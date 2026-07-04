import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from artifacts import build_merkle_manifest, tensor_mapping_sha256


def _tensor_hash(weights: Path) -> Optional[str]:
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None
    return tensor_mapping_sha256(load_file(str(weights), device="cpu"))


def build_model_card(name: str, manifest: Dict[str, Any]) -> str:
    return f"""---
tags:
- openverifiablellm
- model-verification
- sigstore
- merkle-tree
---

# {name}

This model is published with OpenVerifiableLLM verification metadata.

This repo includes:

- safetensors weights
- `ovllm_manifest.json`
- Sigstore/model-transparency bundle
- Merkle metadata

## Verify

```bash
ovllm verify <model-ref>
```

For a local clone of this model repository:

```bash
ovllm verify .
```

The verifier recomputes raw artifact SHA-256, Merkle chunk metadata, and the
safetensors tensor hash, then verifies the Sigstore/model-transparency bundle.

## Artifact Integrity

- weights: `{manifest["weights"]}`
- sha256: `{manifest["sha256"]}`
- Merkle root: `{manifest["merkle_root"]}`
- chunk size: `{manifest["chunk_size_bytes"]}`
- chunks: `{manifest["chunk_count"]}`

## Signature

Sigstore/model-transparency bundle: `{manifest.get("signature", "model.sig")}`
"""


def build_modelfile(name: str, weights_name: str) -> str:
    return f"""# OpenVerifiableLLM Ollama build file
# Replace FROM with the nearest compatible base when publishing a real Ollama artifact.
FROM ./{weights_name}
PARAMETER temperature 0
MESSAGE system "Verified OpenVerifiableLLM artifact: {name}"
"""


def prepare_publish_dir(
    *,
    weights: str,
    output_dir: str,
    name: str = "openverifiable-small",
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    src = Path(weights)
    if not src.exists():
        raise FileNotFoundError(f"weights not found: {src}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

    merkle = build_merkle_manifest(dst)
    manifest = {
        "schema": "ovllm.model.v1",
        "name": name,
        "weights": dst.name,
        "sha256": merkle["sha256"],
        "merkle_root": merkle["merkle_root"],
        "chunk_size_bytes": merkle["chunk_size_bytes"],
        "chunk_count": merkle["chunk_count"],
        "param_sha256": _tensor_hash(dst),
        "signature": "model.sig",
    }
    if name == "gpt10m-shakespeare":
        manifest["segment_replay"] = {
            "model": "gpt10m",
            "dataset": "shakespeare",
            "precision": "fp32",
            "deterministic": True,
            "seed": 99,
            "device": "cpu",
            "overrides": {
                "total_steps": 8,
                "batch_size": 8,
                "block_size": 64
            },
            "expected_param_sha256": manifest["param_sha256"]
        }
    if manifest_extra:
        manifest.update(manifest_extra)

    (out / "ovllm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(build_model_card(name, manifest), encoding="utf-8")
    (out / "Modelfile").write_text(build_modelfile(name, dst.name), encoding="utf-8")
    return out


def _manifest_path(model_dir: Path) -> Path:
    return model_dir / "ovllm_manifest.json"


def _load_manifest(model_dir: Path) -> Dict[str, Any]:
    path = _manifest_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(model_dir: Path, manifest: Dict[str, Any]) -> None:
    _manifest_path(model_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _resolve_signature_path(model_dir: Path, signature: str) -> Path:
    signature_path = Path(signature)
    if not signature_path.is_absolute():
        signature_path = model_dir / signature_path
    return signature_path


def sign_model_dir(
    model_dir: str,
    signature: str = "model.sig",
    *,
    identity: Optional[str] = None,
    identity_provider: Optional[str] = None,
    use_ambient_credentials: bool = False,
    dry_run: bool = False,
) -> int:
    model_path = Path(model_dir).resolve()
    signature_path = _resolve_signature_path(model_path, signature).resolve()
    if not signature_path.is_relative_to(model_path):
        raise ValueError(f"signature path must be inside model directory: {signature_path}")
    signature_path.parent.mkdir(parents=True, exist_ok=True)

    import os
    if not dry_run and os.environ.get("GITHUB_ACTIONS") != "true" and os.environ.get("OVLLM_ALLOW_LOCAL_SIGNING") != "true":
        print(
            "Error: Local signing is disabled by default because it expects GitHub Actions.\n"
            "The project-scoped signing path should use the GitHub Actions workflow.\n"
            "To bypass this check for local testing, set the environment variable: OVLLM_ALLOW_LOCAL_SIGNING=true",
            file=sys.stderr
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "model_signing",
        "sign",
        "sigstore",
        str(model_path),
        "--signature",
        str(signature_path),
        "--ignore-paths",
        str(signature_path),
    ]
    if use_ambient_credentials:
        cmd.append("--use_ambient_credentials")
    if dry_run:
        print(" ".join(cmd))
        return 0

    manifest = _load_manifest(model_path)
    original_manifest = dict(manifest)
    manifest["signature"] = signature_path.name if signature_path.parent == model_path else str(signature_path)
    if identity:
        manifest["sigstore_identity"] = identity
    if identity_provider:
        manifest["sigstore_identity_provider"] = identity_provider
    manifest["sigstore_signed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_manifest(model_path, manifest)

    code = subprocess.run(cmd, check=False).returncode
    if code != 0:
        _write_manifest(model_path, original_manifest)
    return code


def publish_huggingface(repo_id: str, model_dir: str, *, dry_run: bool = False) -> int:
    if dry_run:
        print(f"Native HF Upload: {model_dir} -> {repo_id}")
        return 0

    try:
        import importlib
        import os

        # These verifier artifacts are small; disabling Xet avoids Windows cache
        # permission failures in ~/.cache/huggingface/xet during manual uploads.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        hf_hub = importlib.import_module("huggingface_hub")
        HfApi = getattr(hf_hub, "HfApi")
    except ImportError as e:
        print("huggingface_hub is not installed. Install it to publish to Hugging Face.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error importing Hugging Face Hub client: {e}", file=sys.stderr)
        return 1

    try:
        # Pulls the ambient token mapped into the workflow's environment block
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        # Automatically creates the model repo if it doesn't exist yet
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Uploads your directory containing weights, signatures, and manifest
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model"
        )
        return 0
    except Exception as e:
        print(f"Error publishing to Hugging Face natively: {e}", file=sys.stderr)
        return 1


def build_ollama(model_name: str, model_dir: str, *, dry_run: bool = False) -> int:
    cmd = ["ollama", "create", model_name, "-f", str(Path(model_dir) / "Modelfile")]
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, check=False).returncode
