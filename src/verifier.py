import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from artifacts import (
    MERKLE_ALG_LEGACY,
    MERKLE_ALGS,
    build_merkle_manifest,
    compute_sha256,
    tensor_mapping_sha256,
)


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    expected: Optional[str] = None
    actual: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in {PASS, SKIP}


def resolve_model_reference(ref: str, cache_dir: Optional[str] = None) -> Path:
    path = Path(ref)
    if path.exists():
        return path.resolve()

    try:
        hf_hub = importlib.import_module("huggingface_hub")
        snapshot_download = hf_hub.snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"Model reference {ref!r} is not a local path and huggingface_hub is not installed. "
            "Install it with `pip install huggingface_hub` or pass a local directory."
        ) from exc

    if cache_dir is None:
        cache_dir = os.environ.get("OVLLM_HF_CACHE_DIR") or str(
            Path.cwd() / ".ovllm-cache" / "huggingface"
        )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    return Path(
        snapshot_download(
            repo_id=ref,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.safetensors",
                "ovllm_manifest.json",
                "pipeline_manifest.json",
                "model.sig",
                "*.sig",
                "*.bundle",
                "*.json",
                "README.md",
                "Modelfile",
            ],
        )
    ).resolve()


def load_manifest(model_dir: Path, manifest_name: str = "ovllm_manifest.json") -> Dict[str, Any]:
    path = model_dir / manifest_name
    if not path.exists():
        alt = model_dir / "pipeline_manifest.json"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Missing manifest: expected {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["_manifest_path"] = str(path)
    return manifest


def _first_existing(model_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = model_dir / name
        if path.exists():
            return path
    return None


def find_weights(model_dir: Path, manifest: Dict[str, Any]) -> Path:
    declared = manifest.get("weights") or manifest.get("model_checkpoint_artifact")
    if declared:
        path = model_dir / declared
        if path.exists():
            return path
        raise FileNotFoundError(f"Manifest declares missing weights artifact: {path}")

    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No .safetensors weights found in {model_dir}")
    if len(weights) > 1:
        raise RuntimeError(
            f"Multiple .safetensors files found in {model_dir}; set manifest['weights'] explicitly."
        )
    return weights[0]


def _check_equal(name: str, expected: Optional[str], actual: str, detail: str) -> CheckResult:
    if expected is None:
        return CheckResult(name, SKIP, f"{detail}; no expected value in manifest", actual=actual)
    return CheckResult(
        name,
        PASS if expected == actual else FAIL,
        detail,
        expected=expected,
        actual=actual,
    )


def check_artifact_hashes(model_dir: Path, manifest: Dict[str, Any]) -> List[CheckResult]:
    results: List[CheckResult] = []
    weights = find_weights(model_dir, manifest)

    manifest_sha = (
        manifest.get("sha256")
        or manifest.get("weights_sha256")
        or manifest.get("model_checkpoint_hash")
        or manifest.get("4_model_checkpoint_hash")
    )
    results.append(
        _check_equal(
            "artifact_sha256",
            manifest_sha,
            compute_sha256(file_path=weights),
            f"raw bytes of {weights.name}",
        )
    )

    # A manifest with no merkle_alg predates the RFC 6962 hardening, so its root
    # was built with the legacy construction. Recompute with the algorithm the
    # manifest was actually written under -- otherwise an untampered artifact
    # reports a root mismatch, which reads as tampering.
    # NOTE: An absent key defaults to legacy; an explicit None is unknown/invalid.
    alg_key_present = "merkle_alg" in manifest
    declared_alg = manifest.get("merkle_alg")
    alg_known = (not alg_key_present) or (declared_alg in MERKLE_ALGS)
    merkle_alg = declared_alg if declared_alg in MERKLE_ALGS else MERKLE_ALG_LEGACY

    # chunk_count and sha256 are independent of the tree construction, so this
    # is safe to build even when the declared algorithm is unrecognised.
    merkle = build_merkle_manifest(weights, alg=merkle_alg)

    if not alg_known:
        results.append(
            CheckResult(
                "merkle_root",
                FAIL,
                f"manifest declares unknown merkle_alg {declared_alg!r}",
                expected=f"one of {', '.join(MERKLE_ALGS)}",
                actual=str(declared_alg),
            )
        )
    else:
        expected_root = (
            manifest.get("merkle_root")
            or manifest.get("weights_merkle_root")
            or manifest.get("4_model_checkpoint_merkle_root")
        )
        detail = f"1 MB chunk tree ({merkle_alg})"
        if not alg_key_present:
            detail += " -- manifest predates merkle_alg; assumed legacy"
        results.append(
            _check_equal("merkle_root", expected_root, merkle["merkle_root"], detail)
        )

    expected_chunks = (
        manifest.get("chunk_count")
        or manifest.get("weights_chunk_count")
        or manifest.get("4_model_checkpoint_chunk_count")
    )
    if expected_chunks is None:
        results.append(
            CheckResult("merkle_chunk_count", SKIP, "no expected chunk count in manifest")
        )
    else:
        results.append(
            CheckResult(
                "merkle_chunk_count",
                PASS if int(expected_chunks) == merkle["chunk_count"] else FAIL,
                "number of chunks in Merkle manifest",
                expected=str(expected_chunks),
                actual=str(merkle["chunk_count"]),
            )
        )

    expected_tensor = manifest.get("param_sha256") or manifest.get("tensor_sha256")
    try:
        from safetensors.torch import load_file

        tensor_hash = tensor_mapping_sha256(load_file(str(weights), device="cpu"))
        results.append(
            _check_equal("tensor_sha256", expected_tensor, tensor_hash, "safetensors tensor bytes")
        )
    except Exception as exc:
        results.append(CheckResult("tensor_sha256", FAIL, f"could not read safetensors: {exc}"))

    return results


def check_sigstore_bundle(
    model_dir: Path,
    manifest: Dict[str, Any],
    *,
    allow_unsigned: bool = False,
) -> CheckResult:
    # Resolve symlinks in the base directory up front
    model_dir = Path(model_dir).resolve()

    signature = manifest.get("signature") or manifest.get("sigstore_bundle")
    signature_path = model_dir / signature if signature else _first_existing(
        model_dir, ["model.sig", "model.bundle", "sigstore.bundle"]
    )
    
    if signature_path is None or not signature_path.exists():
        status = SKIP if allow_unsigned else FAIL
        return CheckResult("sigstore_bundle", status, "missing Sigstore/model-signing bundle")

    # Resolve symlinks for the specific signature file path
    signature_path = signature_path.resolve()

    identity = manifest.get("sigstore_identity")
    provider = manifest.get("sigstore_identity_provider")
    if not identity or not provider:
        status = SKIP if allow_unsigned else FAIL
        return CheckResult(
            "sigstore_bundle",
            status,
            "signature present, but manifest lacks sigstore_identity/provider",
        )

    # Verify model_signing is installed in the active python environment
    import importlib.util
    if importlib.util.find_spec("model_signing") is None:
        return CheckResult(
            "sigstore_bundle",
            FAIL,
            f"model_signing module is not installed in the active environment ({sys.executable}). "
            "Please install it with `pip install model-signing` to enable Sigstore bundle verification.",
        )

    cmd = [
        sys.executable,
        "-m",
        "model_signing",
        "verify",
        "sigstore",
        str(model_dir),
        "--signature",
        str(signature_path),
        "--ignore-paths",
        str(signature_path),
        "--identity",
        identity,
        "--identity_provider",
        provider,
        "--allow_symlinks",
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except ModuleNotFoundError as exc:
        return CheckResult("sigstore_bundle", FAIL, f"model_signing is not installed: {exc}")
    except OSError as exc:
        return CheckResult("sigstore_bundle", FAIL, f"could not launch model_signing: {exc}")

    detail = (completed.stdout or completed.stderr).strip()
    return CheckResult(
        "sigstore_bundle",
        PASS if completed.returncode == 0 else FAIL,
        detail or "model_signing verify completed",
    )


def check_segment_replay(manifest: Dict[str, Any], *, skip_replay: bool = False) -> CheckResult:
    spec = manifest.get("segment_replay")
    if skip_replay:
        return CheckResult("segment_replay", SKIP, "disabled by --skip-replay")
    if not spec:
        return CheckResult("segment_replay", SKIP, "manifest has no segment_replay spec")

    try:
        from experiment import run_one

        record = run_one(
            spec["model"],
            spec.get("dataset", "shakespeare"),
            spec.get("precision", "fp32"),
            spec.get("deterministic", True),
            spec.get("seed", 99),
            device=spec.get("device", "cpu"),
            overrides=spec.get("overrides", {}),
            track_full=False,
            keep_artifact=False,
            twin=False,
            quiet=True,
        )
    except Exception as exc:
        return CheckResult("segment_replay", FAIL, f"replay failed: {exc}")

    expected = spec.get("expected_param_sha256") or spec.get("param_sha256")
    if expected is None:
        return CheckResult(
            "segment_replay",
            SKIP,
            "replay ran, but no expected_param_sha256 was provided",
            actual=record["param_sha256"],
        )
    return CheckResult(
        "segment_replay",
        PASS if expected == record["param_sha256"] else FAIL,
        "sampled deterministic replay window",
        expected=expected,
        actual=record["param_sha256"],
    )


def verify_model_reference(
    ref: str,
    *,
    cache_dir: Optional[str] = None,
    allow_unsigned: bool = False,
    skip_replay: bool = False,
) -> List[CheckResult]:
    model_dir = resolve_model_reference(ref, cache_dir=cache_dir)
    manifest = load_manifest(model_dir)
    results = [
        CheckResult("resolve_reference", PASS, str(model_dir)),
        CheckResult("manifest_json", PASS, manifest["_manifest_path"]),
    ]
    results.extend(check_artifact_hashes(model_dir, manifest))
    results.append(check_segment_replay(manifest, skip_replay=skip_replay))
    results.append(check_sigstore_bundle(model_dir, manifest, allow_unsigned=allow_unsigned))
    return results


def print_report(results: List[CheckResult]) -> bool:
    for result in results:
        suffix = f" - {result.detail}" if result.detail else ""
        print(f"[{result.status:<4}] {result.name}{suffix}")
        if result.expected is not None or result.actual is not None:
            print(f"       expected: {result.expected}")
            print(f"       actual  : {result.actual}")
    ok = all(result.ok for result in results)
    print("\nVERDICT:", "GREEN" if ok else "RED")
    return ok
