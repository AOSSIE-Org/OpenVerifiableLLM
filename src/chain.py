"""Checkpoint-chain trainer and k-of-N sampled segment auditor.

This implements the audit mechanism described in the README's "What a
spot-check audit buys": the single mid-checkpoint replay in reproducibility.py
proves the mechanism for one segment; this module runs it at chain scale.

  * ``train``: a prover trains N segments and seals a SIGNED full-training-state
    boundary checkpoint (weights, optimizer, all four RNG streams) at every
    boundary, starting from boundary 0 -- the deterministically seeded init --
    so even the first segment is verifiable. A signed ``chain_manifest.json``
    records the commitment (dataset hash, full config, seed) and per-segment
    wall times. In a real deployment the commitment is published to a
    transparency log BEFORE training; here it is written alongside the chain.

  * ``audit``: an auditor samples k of the N segments uniformly (seeded, so an
    audit is itself reproducible), replays each from its opening boundary, and
    compares the closing parameter hash BIT-EXACTLY against the manifest. No
    tolerance window: a segment either reproduces the exact hash or fails.
    The report measures rather than asserts the economics: auditor wall time
    vs. prover wall time (the realized k/N cost ratio) and the detection
    probability k/N that this sample size buys against a minimal
    single-bad-segment forgery.

CPU smoke (~a minute):

    python chain.py train --model mlp --segments 4 --segment-steps 3 \
        --batch-size 4 --block-size 48
    python chain.py audit --k 2

Pod scale: see RUNBOOK.md section 8.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from artifacts import RUNS_DIR, hash_json, model_parameters_sha256
from config import model_config
from dataset import get_dataset
from device import accel_rng_state, device_name, get_device, restore_accel_rng_state
from main import set_seed
from model import build_model, count_params
from signing import SignatureError, sign_file, signed_torch_save, verified_torch_load, verify_file

CHAIN_MANIFEST_NAME = "chain_manifest.json"
DEFAULT_CHAIN_DIR = RUNS_DIR / "chain"


def _boundary_name(index):
    return f"boundary_{index:04d}.pt"


def _dataset_sha256(dataset):
    return hashlib.sha256(dataset.encoded.numpy().tobytes()).hexdigest()


def _resolve_device(device):
    return get_device() if device in (None, "auto") else torch.device(device)


def _train_steps(model, optimizer, dataset, steps, batch_size, block_size, dev):
    for _ in range(steps):
        # FIRST statement in the loop: the batch draw consumes the global torch
        # RNG, which the boundary checkpoint saves/restores -- this is what keeps
        # segment replay bitwise-exact.
        x, y = dataset.get_batch(batch_size, block_size, device=dev)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def _save_boundary(chain_dir, index, step, model, optimizer, wall_time_s):
    path = chain_dir / _boundary_name(index)
    param_hash = model_parameters_sha256(model)
    signed_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "accel_rng_state": accel_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "step": step,
            "boundary_index": index,
            "param_sha256": param_hash,
        },
        path,
    )
    return {
        "index": index,
        "step": step,
        "param_sha256": param_hash,
        "file": path.name,
        "segment_wall_time_s": round(wall_time_s, 2),
    }


def train_chain(model_name="gpt10m", dataset_name="shakespeare",
                out_dir=DEFAULT_CHAIN_DIR, num_segments=10, segment_steps=20,
                seed=None, device="auto", overrides=None):
    """Train N segments, sealing a signed boundary checkpoint at each boundary.

    Returns the chain manifest (also written, signed, to out_dir).
    """
    dev = _resolve_device(device)
    cfg = model_config(model_name, **(overrides or {}))
    seed = seed if seed is not None else cfg["seed"]
    chain_dir = Path(out_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    dataset = get_dataset(dataset_name, block_size=cfg["block_size"])
    model = build_model(model_name, dataset.vocab_size, cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    print(f"[chain] {model_name} ({count_params(model):,} params) on {dataset_name} | "
          f"{num_segments} segments x {segment_steps} steps | {dev.type} ({device_name(dev)})")

    # Boundary 0: the deterministically seeded init, before any training step.
    boundaries = [_save_boundary(chain_dir, 0, 0, model, optimizer, 0.0)]

    bs, blk = cfg["batch_size"], cfg["block_size"]
    t_start = time.time()
    for seg in range(num_segments):
        t0 = time.time()
        loss = _train_steps(model, optimizer, dataset, segment_steps, bs, blk, dev)
        boundaries.append(
            _save_boundary(chain_dir, seg + 1, (seg + 1) * segment_steps,
                           model, optimizer, time.time() - t0))
        print(f"  segment {seg:3d} sealed | loss {loss:.6f} | "
              f"hash {boundaries[-1]['param_sha256'][:12]} | "
              f"{boundaries[-1]['segment_wall_time_s']}s")
    prover_wall = time.time() - t_start

    manifest = {
        # Everything that determines the trajectory. In a real deployment this
        # block is published to a transparency log BEFORE training starts.
        "commitment": {
            "model": model_name,
            "dataset": dataset_name,
            "dataset_sha256": _dataset_sha256(dataset),
            "config": cfg,
            "config_sha256": hash_json(cfg),
            "seed": seed,
            "num_segments": num_segments,
            "segment_steps": segment_steps,
        },
        "environment": {
            "torch": torch.__version__,
            "device": dev.type,
            "device_name": device_name(dev),
        },
        "num_params": count_params(model),
        "final_param_sha256": boundaries[-1]["param_sha256"],
        "prover_wall_time_s": round(prover_wall, 2),
        "boundaries": boundaries,
    }
    manifest_path = chain_dir / CHAIN_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sign_file(manifest_path)
    print(f"[chain] sealed {num_segments}-segment chain in {prover_wall:.1f}s -> {chain_dir}")
    return manifest


def _replay_segment(chain_dir, manifest, seg_index, dev):
    """Replay one segment from its opening boundary; return the result record."""
    c = manifest["commitment"]
    cfg = c["config"]
    opening = manifest["boundaries"][seg_index]
    closing = manifest["boundaries"][seg_index + 1]

    t0 = time.time()
    result = {"segment": seg_index, "expected": closing["param_sha256"]}
    try:
        # Fresh scaffolding; every piece of state is then overwritten from the
        # verified boundary checkpoint.
        set_seed(c["seed"])
        dataset = get_dataset(c["dataset"], block_size=cfg["block_size"])
        model = build_model(c["model"], dataset.vocab_size, cfg).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        ckpt = verified_torch_load(chain_dir / opening["file"], map_location=dev)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng_state"].cpu())
        restore_accel_rng_state(ckpt.get("accel_rng_state"))
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])

        if model_parameters_sha256(model) != opening["param_sha256"]:
            result.update(ok=False, reason="opening boundary hash mismatch")
            return result

        _train_steps(model, optimizer, dataset, c["segment_steps"],
                     cfg["batch_size"], cfg["block_size"], dev)
        got = model_parameters_sha256(model)
        result.update(got=got, ok=(got == closing["param_sha256"]),
                      reason=None if got == closing["param_sha256"] else "closing hash mismatch")
    except SignatureError as exc:
        result.update(ok=False, reason=f"signature: {exc}")
    finally:
        result["wall_time_s"] = round(time.time() - t0, 2)
    return result


def audit_chain(chain_dir=DEFAULT_CHAIN_DIR, k=3, audit_seed=0, device="auto",
                segments=None):
    """Sample k of N segments, replay each bit-exactly, report the economics.

    Returns a report dict; ``report["ok"]`` is True iff every sampled segment
    (and the dataset commitment) verified.
    """
    chain_dir = Path(chain_dir)
    dev = _resolve_device(device)

    manifest_path = chain_dir / CHAIN_MANIFEST_NAME
    verify_file(manifest_path)  # signature over the manifest itself, first
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    c = manifest["commitment"]
    n = c["num_segments"]

    if segments is None:
        segments = sorted(random.Random(audit_seed).sample(range(n), min(k, n)))
    k = len(segments)

    print(f"[audit] chain of {n} segments | sampling {k}: {segments} "
          f"(audit_seed={audit_seed}) | {dev.type} ({device_name(dev)})")

    # Dataset substitution check: the corpus we would replay with must match
    # the committed hash, or every replay would be meaningless.
    dataset = get_dataset(c["dataset"], block_size=c["config"]["block_size"])
    dataset_ok = _dataset_sha256(dataset) == c["dataset_sha256"]
    if not dataset_ok:
        print("[audit] FAIL: local dataset does not match the committed dataset hash")

    results = [_replay_segment(chain_dir, manifest, s, dev) for s in segments]
    for r in results:
        print(f"  segment {r['segment']:3d} | {'PASS' if r['ok'] else 'FAIL'}"
              f"{'' if r['ok'] else ' (' + r['reason'] + ')'} | {r['wall_time_s']}s")

    audit_wall = round(sum(r["wall_time_s"] for r in results), 2)
    prover_wall = manifest["prover_wall_time_s"]
    ratio = round(audit_wall / prover_wall, 4) if prover_wall else None
    detection = round(k / n, 4)

    report = {
        "chain_dir": str(chain_dir),
        "num_segments": n,
        "sampled_segments": segments,
        "k": k,
        "dataset_ok": dataset_ok,
        "results": results,
        "audit_wall_time_s": audit_wall,
        "prover_wall_time_s": prover_wall,
        "cost_ratio": ratio,
        "min_forgery_detection_probability": detection,
        "ok": dataset_ok and all(r["ok"] for r in results),
    }

    print(f"[audit] {'PASS' if report['ok'] else 'FAIL'} | "
          f"auditor {audit_wall}s vs prover {prover_wall}s "
          f"(measured cost ratio {ratio}) | "
          f"a minimal single-bad-segment forgery is caught with p = k/N = {detection}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train an N-segment signed checkpoint chain")
    t.add_argument("--model", default="gpt10m")
    t.add_argument("--dataset", default="shakespeare")
    t.add_argument("--segments", type=int, default=10)
    t.add_argument("--segment-steps", type=int, default=20)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--device", default="auto")
    t.add_argument("--out", default=str(DEFAULT_CHAIN_DIR))
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--block-size", type=int, default=None)

    a = sub.add_parser("audit", help="sample k segments and replay them bit-exactly")
    a.add_argument("chain_dir", nargs="?", default=str(DEFAULT_CHAIN_DIR))
    a.add_argument("--k", type=int, default=3)
    a.add_argument("--audit-seed", type=int, default=0)
    a.add_argument("--device", default="auto")
    a.add_argument("--segments", default=None,
                   help="comma-separated explicit segment indices (overrides --k)")

    args = p.parse_args()
    if args.cmd == "train":
        overrides = {}
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        if args.block_size:
            overrides["block_size"] = args.block_size
        train_chain(args.model, args.dataset, out_dir=args.out,
                    num_segments=args.segments, segment_steps=args.segment_steps,
                    seed=args.seed, device=args.device, overrides=overrides)
    else:
        segments = ([int(s) for s in args.segments.split(",")]
                    if args.segments else None)
        report = audit_chain(args.chain_dir, k=args.k, audit_seed=args.audit_seed,
                             device=args.device, segments=segments)
        sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
