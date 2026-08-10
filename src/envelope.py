"""Determinism envelope: does bit-exactness survive the optimizations real
training uses -- fused SDPA attention backends and torch.compile?

For each execution variant, two questions (the same separation the matrix
uses): run-to-run reproducibility (twin runs, same seed, same device) and
bitwise agreement with the eager/manual reference. Each cell runs under
strict determinism ON and OFF; kernels that strict mode refuses are recorded
as errors -- that refusal is itself a datum. Dropout is forced to 0.0 so the
comparison isolates arithmetic, not RNG-stream layout differences between
attention implementations.

The last exhibit is the one that matters for the audit story: a checkpoint
chain trained WITH torch.compile + flash SDPA (the variant committed in its
config), then segment-audited -- does bit-exact replay survive compilation?

    python envelope.py --device cuda --steps 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from artifacts import RUNS_DIR, model_parameters_sha256
from chain import audit_chain, train_chain
from config import model_config
from dataset import get_dataset
from device import device_name, get_device, sdpa_context
from main import set_seed
from model import build_model

VARIANTS = [
    # (name, attn_impl, sdpa_backend, compile)
    ("eager_manual", "manual", None, False),
    ("eager_sdpa_math", "sdpa", "math", False),
    ("eager_sdpa_flash", "sdpa", "flash", False),
    ("eager_sdpa_efficient", "sdpa", "efficient", False),
    ("compiled_manual", "manual", None, True),
    ("compiled_sdpa_flash", "sdpa", "flash", True),
]


def _train(model_name, dataset_name, cfg, attn_impl, sdpa_backend, compiled,
           deterministic, dev, steps):
    set_seed(cfg["seed"], deterministic=deterministic, warn_only=False)
    ds = get_dataset(dataset_name, block_size=cfg["block_size"])
    cfg = dict(cfg, attn_impl=attn_impl)
    model = build_model(model_name, ds.vocab_size, cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    fwd = torch.compile(model) if compiled else model
    with sdpa_context(sdpa_backend):
        for _ in range(steps):
            x, y = ds.get_batch(cfg["batch_size"], cfg["block_size"], device=dev)
            logits = fwd(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model_parameters_sha256(model), loss.item()


def run_envelope(model_name="gpt10m", dataset_name="shakespeare", steps=300,
                 device="auto", out_path=RUNS_DIR / "envelope_report.json"):
    dev = get_device() if device in (None, "auto") else torch.device(device)
    cfg = model_config(model_name, dropout=0.0)

    cells = []
    reference = {}
    for name, attn, backend, compiled in VARIANTS:
        for det in (True, False):
            cell = {"variant": name, "deterministic": det}
            t0 = time.time()
            try:
                h1, l1 = _train(model_name, dataset_name, cfg, attn, backend,
                                compiled, det, dev, steps)
                h2, _ = _train(model_name, dataset_name, cfg, attn, backend,
                               compiled, det, dev, steps)
                cell.update(ok=True, reproducible=(h1 == h2),
                            param_sha256=h1, final_loss=l1)
                if name == "eager_manual" and det:
                    reference["hash"] = h1
                cell["vs_eager_manual_bitwise"] = (
                    h1 == reference.get("hash") if reference else None)
            except Exception as exc:  # strict-mode kernel refusals land here
                cell.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
            cell["wall_time_s"] = round(time.time() - t0, 2)
            cells.append(cell)
            status = ("repro=" + str(cell.get("reproducible"))
                      + " vs_ref=" + str(cell.get("vs_eager_manual_bitwise"))
                      if cell["ok"] else "ERROR " + cell["error"].split(":")[0])
            print(f"  [{name:<22} det={'on ' if det else 'off'}] {status} "
                  f"({cell['wall_time_s']}s)")

    # The audit-story exhibit: a chain trained under compile+flash, then audited.
    chain_cell = {"name": "compiled_flash_chain_audit"}
    try:
        overrides = dict(dropout=0.0, attn_impl="sdpa", sdpa_backend="flash",
                         compile=True)
        train_chain(model_name, dataset_name, out_dir=RUNS_DIR / "envelope_chain",
                    num_segments=3, segment_steps=max(steps // 3, 1),
                    device=device, overrides=overrides)
        report = audit_chain(RUNS_DIR / "envelope_chain", k=3, device=device)
        chain_cell.update(ok=True, audit_pass=report["ok"],
                          results=[{k: r[k] for k in ("segment", "ok", "reason")}
                                   for r in report["results"]])
    except Exception as exc:
        chain_cell.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
    print(f"  [chain audit under compile+flash] "
          f"{chain_cell.get('audit_pass', chain_cell.get('error'))}")

    out = {
        "model": model_name, "dataset": dataset_name, "steps": steps,
        "device": device_name(dev), "torch": torch.__version__,
        "cells": cells, "compiled_chain_audit": chain_cell,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[envelope] -> {out_path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt10m")
    p.add_argument("--dataset", default="shakespeare")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(RUNS_DIR / "envelope_report.json"))
    args = p.parse_args()
    report = run_envelope(args.model, args.dataset, steps=args.steps,
                          device=args.device, out_path=args.out)
    sys.exit(0 if all(c["ok"] or True for c in report["cells"]) else 1)


if __name__ == "__main__":
    main()
