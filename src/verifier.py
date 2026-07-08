import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from artifacts import build_merkle_manifest, compute_sha256, tensor_mapping_sha256


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

    merkle = build_merkle_manifest(weights)
    expected_root = (
        manifest.get("merkle_root")
        or manifest.get("weights_merkle_root")
        or manifest.get("4_model_checkpoint_merkle_root")
    )
    results.append(
        _check_equal("merkle_root", expected_root, merkle["merkle_root"], "1 MB chunk tree")
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


def get_system_diagnostics() -> Dict[str, str]:
    diag = {
        "OS": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python Version": platform.python_version(),
    }
    try:
        import torch
        diag["PyTorch Version"] = torch.__version__
        if torch.cuda.is_available():
            diag["Accelerator"] = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            diag["Accelerator"] = f"XPU ({torch.xpu.get_device_name(0)})"
        else:
            diag["Accelerator"] = "CPU"
    except ImportError:
        diag["PyTorch Version"] = "Not Installed"
        diag["Accelerator"] = "N/A"
    return diag


def generate_markdown_report(results: List[CheckResult], ok: bool, ref: str, diag: Dict[str, str]) -> str:
    verdict_str = "🟢 **GREEN** (Passed)" if ok else "🔴 **RED** (Failed)"
    md = []
    md.append("# OpenVerifiableLLM Verification Report\n")
    md.append(f"**Verdict:** {verdict_str}\n")
    md.append(f"- **Model Reference:** `{ref}`")
    md.append(f"- **Timestamp:** `{time_ref()}`\n")
    
    md.append("## Check Results\n")
    md.append("| Status | Check Name | Expected | Actual | Details |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        status_icon = "✅ PASS" if r.status == PASS else ("❌ FAIL" if r.status == FAIL else "⚠️ SKIP")
        expected_val = f"`{r.expected}`" if r.expected else "-"
        actual_val = f"`{r.actual}`" if r.actual else "-"
        detail_val = r.detail.replace("\n", " ") if r.detail else ""
        md.append(f"| {status_icon} | **{r.name}** | {expected_val} | {actual_val} | {detail_val} |")
    
    md.append("\n## System Diagnostics\n")
    for k, v in diag.items():
        md.append(f"- **{k}:** `{v}`")
    
    return "\n".join(md) + "\n"


def generate_html_report(results: List[CheckResult], ok: bool, ref: str, diag: Dict[str, str]) -> str:
    verdict_class = "verdict-green" if ok else "verdict-red"
    verdict_text = "VERDICT: GREEN" if ok else "VERDICT: RED"
    
    rows = []
    for r in results:
        status_class = f"status-{r.status.lower()}"
        expected_val = f"<code>{r.expected}</code>" if r.expected else "-"
        actual_val = f"<code>{r.actual}</code>" if r.actual else "-"
        detail_val = r.detail.replace("\n", "<br>") if r.detail else ""
        rows.append(f"""
        <tr>
            <td><span class="status-badge {status_class}">{r.status}</span></td>
            <td><strong>{r.name}</strong></td>
            <td class="mono">{expected_val}</td>
            <td class="mono">{actual_val}</td>
            <td>{detail_val}</td>
        </tr>
        """)
    rows_html = "\n".join(rows)
    
    diag_items = []
    for k, v in diag.items():
        diag_items.append(f"""
        <div class="diag-item">
            <span class="diag-key">{k}</span>
            <span class="diag-val">{v}</span>
        </div>
        """)
    diag_html = "\n".join(diag_items)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OpenVerifiableLLM Verification Report</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --green: #10b981;
            --red: #ef4444;
            --amber: #f59e0b;
            --border-color: #334155;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 2rem;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 900px;
            width: 100%;
        }}
        header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            margin: 0 0 1rem 0;
            font-size: 2rem;
            font-weight: 700;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            font-size: 0.95rem;
        }}
        .meta-item {{
            display: flex;
            flex-direction: column;
        }}
        .meta-label {{
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        .meta-val {{
            font-weight: 600;
        }}
        .verdict-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 1.1rem;
            text-align: center;
        }}
        .verdict-green {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--green);
            border: 1px solid var(--green);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
        }}
        .verdict-red {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--red);
            border: 1px solid var(--red);
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);
        }}
        .report-section {{
            background-color: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        h2 {{
            margin-top: 0;
            font-size: 1.35rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
            color: var(--text-color);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            text-align: left;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.95rem;
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-align: center;
        }}
        .status-pass {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--green);
            border: 1px solid var(--green);
        }}
        .status-fail {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--red);
            border: 1px solid var(--red);
        }}
        .status-skip {{
            background-color: rgba(245, 158, 11, 0.15);
            color: var(--amber);
            border: 1px solid var(--amber);
        }}
        .mono code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            word-break: break-all;
        }}
        .diag-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        .diag-item {{
            display: flex;
            flex-direction: column;
            background-color: rgba(0, 0, 0, 0.15);
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
        .diag-key {{
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        .diag-val {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OpenVerifiableLLM Report</h1>
            <div class="meta-grid">
                <div class="meta-item">
                    <span class="meta-label">Status</span>
                    <div><span class="verdict-badge {verdict_class}">{verdict_text}</span></div>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Model Reference</span>
                    <span class="meta-val">{ref}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Timestamp</span>
                    <span class="meta-val">{time_ref()}</span>
                </div>
            </div>
        </header>

        <section class="report-section">
            <h2>Verification Check Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Check Name</th>
                        <th>Expected</th>
                        <th>Actual</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </section>

        <section class="report-section">
            <h2>System Diagnostics</h2>
            <div class="diag-grid">
                {diag_html}
            </div>
        </section>
    </div>
</body>
</html>
"""
    return html


def time_ref() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def print_report(
    results: List[CheckResult],
    format_type: str = "text",
    output_path: Optional[str] = None,
    ref: str = "",
) -> bool:
    ok = all(result.ok for result in results)
    
    # 1. Generate text output
    text_lines = []
    for result in results:
        suffix = f" - {result.detail}" if result.detail else ""
        text_lines.append(f"[{result.status:<4}] {result.name}{suffix}")
        if result.expected is not None or result.actual is not None:
            text_lines.append(f"       expected: {result.expected}")
            text_lines.append(f"       actual  : {result.actual}")
    text_lines.append(f"\nVERDICT: {'GREEN' if ok else 'RED'}")
    text_str = "\n".join(text_lines) + "\n"

    # 2. Gather diagnostics
    diag = get_system_diagnostics()

    # 3. Handle reporting depending on format
    if format_type == "text":
        print(text_str, end="")
        if output_path:
            Path(output_path).write_text(text_str, encoding="utf-8")
    elif format_type == "markdown":
        md_str = generate_markdown_report(results, ok, ref, diag)
        if output_path:
            Path(output_path).write_text(md_str, encoding="utf-8")
        else:
            print(md_str, end="")
    elif format_type == "html":
        html_str = generate_html_report(results, ok, ref, diag)
        if output_path:
            Path(output_path).write_text(html_str, encoding="utf-8")
        else:
            print(html_str, end="")
            
    return ok
