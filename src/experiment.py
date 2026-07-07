"""Shared experiment engine for the models x conditions matrix.

ONE parametrized runner. run_experiment.py (single cell), sweep.py (the matrix)
and demo.py (the narrative) all call into here -- the old segmented audit becomes
just one more cell.

Two comparisons are kept deliberately separate, because conflating them is the
single most common reproducibility mistake:

  (A) RUN-TO-RUN reproducibility -- train the SAME config twice on the SAME
      hardware; are the resulting bits identical? Broken by nondeterministic
      kernels (atomicAdd in embedding/scatter backward, cuDNN autotuning) when
      determinism is OFF, and by changing hardware (cross-GPU). This is what
      `reproducible` and `first_divergence_step` measure.

  (B) AGREEMENT-WITH-THE-FP32-REFERENCE -- does a tf32/bf16 run produce the same
      bits (or merely the same loss to a tolerance) as the fp32 reference? TF32
      and bf16 are perfectly run-to-run reproducible yet silently disagree with
      fp32. The sweep computes this against the per-model fp32+deterministic cell;
      it is where the "loss within 1e-6 but hash differs" debate lives.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from device import (
    apply_precision,
    autocast_context,
    configure_determinism,
    device_name,
    get_device,
    precision_flags,
    seed_accelerators,
)
from model import build_model, count_params, is_vision_model
from dataset import get_dataset
from config import model_config
from artifacts import build_merkle_manifest, model_parameters_sha256, save_model_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

REFERENCE_LOSS_RTOL = 1e-6  # verify()'s telemetry diagnostic tolerance (the verdict is the exact hash)


def _norm_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "on", "true", "yes", "y")


def prepare_run(seed, precision, deterministic, warn_only=True):
    """Seed everything, then set determinism and precision for one run."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_accelerators(seed)
    configure_determinism(enabled=deterministic, warn_only=warn_only)
    apply_precision(precision)


def _single_train(model_name, dataset_name, precision, deterministic, seed, dev, cfg,
                  warn_only=True, track_full=False):
    """One full training run. Returns (model, losses, final_param_hash, step_hashes)."""
    prepare_run(seed, precision, deterministic, warn_only)

    ds = get_dataset(dataset_name, block_size=cfg["block_size"])
    model = build_model(model_name, ds.vocab_size, cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    vision = is_vision_model(model_name)
    autocast = autocast_context(precision, dev.type)

    bs, blk = cfg["batch_size"], cfg["block_size"]
    losses, step_hashes = [], []
    for _step in range(cfg["total_steps"]):
        # FIRST statement in the loop: the batch draw consumes the global torch
        # RNG, so it is captured by torch.get_rng_state() and stays replay- and
        # run-to-run exact. Moving this out of the loop silently breaks both.
        x, y = ds.get_batch(bs, blk, device=dev)
        with autocast:
            logits = model(x)
            if vision:
                loss = F.cross_entropy(logits, y)
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if track_full:
            step_hashes.append(model_parameters_sha256(model))

    return model, losses, model_parameters_sha256(model), step_hashes


def _first_divergence(losses_a, losses_b, hashes_a=None, hashes_b=None):
    """First step where two twin runs diverge. Prefer per-step param hashes
    (true bitwise divergence); fall back to exact per-step loss comparison."""
    if hashes_a and hashes_b:
        for i, (ha, hb) in enumerate(zip(hashes_a, hashes_b)):
            if ha != hb:
                return i
        return None
    for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
        if la != lb:
            return i
    return None


def _merkle_from_model(model, tag, keep_artifact):
    """Save safetensors, build the Merkle manifest, return (root, chunk_count).

    The whole point of scaling the model: a ~10M-param checkpoint is ~43 MB, so
    the 1 MB-chunk Merkle tree has ~43 leaves instead of the toy's single chunk.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{tag}.safetensors"
    try:
        save_model_safetensors(model, path)
        manifest = build_merkle_manifest(path)
        return manifest["merkle_root"], manifest["chunk_count"], manifest["size_bytes"]
    finally:
        if not keep_artifact and path.exists():
            path.unlink()


def run_one(model_name, dataset_name="shakespeare", precision="fp32",
            deterministic=True, seed=99, device="auto", overrides=None,
            track_full=False, keep_artifact=False, twin=True, quiet=False):
    """Run a single matrix cell and return its JSON-serializable record."""
    deterministic = _norm_bool(deterministic)
    dev = get_device() if device in (None, "auto") else torch.device(device)
    cfg = model_config(model_name, **(overrides or {}))

    # Validate total_steps
    if cfg.get("total_steps", 0) < 1:
        raise ValueError(f"total_steps must be at least 1, got {cfg.get('total_steps')}")

    tag = f"{model_name}_{dataset_name}_{precision}_det{'on' if deterministic else 'off'}_s{seed}"

    t0 = time.time()
    modelA, lossesA, hashA, stepA = _single_train(
        model_name, dataset_name, precision, deterministic, seed, dev, cfg,
        track_full=track_full)

    reproducible, first_div = None, None
    if twin:
        # Twin run: identical settings, same seed, same hardware -> tests (A).
        _modelB, lossesB, hashB, stepB = _single_train(
            model_name, dataset_name, precision, deterministic, seed, dev, cfg,
            track_full=track_full)
        reproducible = (hashA == hashB) and (lossesA == lossesB)
        first_div = _first_divergence(lossesA, lossesB, stepA, stepB)

    merkle_root, chunk_count, size_bytes = _merkle_from_model(modelA, tag, keep_artifact)

    record = {
        "model": model_name,
        "dataset": dataset_name,
        "precision": precision,
        "deterministic": deterministic,
        "device": dev.type,
        "device_name": device_name(dev),
        "final_loss": lossesA[-1],
        "param_sha256": hashA,
        "merkle_root": merkle_root,
        "merkle_chunk_count": chunk_count,
        "artifact_size_bytes": size_bytes,
        "first_divergence_step": first_div,
        "reproducible": reproducible,
        "num_params": count_params(modelA),
        "seed": seed,
        "total_steps": cfg["total_steps"],
        "batch_size": cfg["batch_size"],
        "block_size": cfg["block_size"],
        "torch": torch.__version__,
        "precision_flags": precision_flags(),
        "wall_time_s": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if track_full:
        record["per_step_param_sha256_runA"] = stepA
        if twin:
            record["per_step_param_sha256_runB"] = stepB
    if not quiet:
        _print_cell(record)
    return record


def _print_cell(r):
    repro = "PASS" if r["reproducible"] else ("FAIL" if r["reproducible"] is not None else "-")
    fd = r["first_divergence_step"]
    print(
        f"  [{r['model']:>6} | {r['dataset']:>11} | {r['precision']:>4} | "
        f"det {'on ' if r['deterministic'] else 'off'} | {r['device']:>4}]  "
        f"loss={r['final_loss']:.6f}  repro={repro}  "
        f"first_div={'-' if fd is None else fd}  "
        f"hash={r['param_sha256'][:12]}  merkle={r['merkle_root'][:12]} "
        f"({r['merkle_chunk_count']} chunks)"
    )


def append_record(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        import json
        f.write(json.dumps(record) + "\n")
