"""Forgery constructors and replay-free detectors for checkpoint chains.

The pillar-4 experiment. Under bit-exact determinism with committed inputs,
any forged chain contains at least one segment that fails replay, and a random
k-of-N audit catches it with probability k/N (README, "What a spot-check audit
buys"). This module measures how much of the remaining 1-k/N gap CHEAP,
REPLAY-FREE statistics close: detectors that read only consecutive boundary
checkpoints and flag transitions that the committed optimizer could not have
produced. A flagged segment is replayed first, so detection quality converts
directly into audit efficiency.

Why this is possible here and not in classic proof-of-learning: chain
boundaries are FULL training states. Each carries Adam's moment estimates, so
a forged weight jump must also be consistent with the gradient history the
moments claim -- a much harder object to forge than the weights alone.

Detectors (all O(params) tensor math -- no training, no replay):

  * reach  -- Adam reachability. Adam's per-coordinate step magnitude is
              bounded (|update| ~ lr * |m_hat|/(sqrt(v_hat)+eps), empirically
              a small multiple of lr), so over S steps no coordinate can move
              much further than S * lr * c. max|dW| / (S * lr) beyond the
              genuine range is unreachable for the committed optimizer.
  * norm   -- ||dW_k|| profile across segments (z-score). Training decays
              smoothly; a splice jumps.
  * moment -- cos(dW_k, -exp_avg_{k+1}). Adam updates follow -m_hat, so the
              segment delta anti-aligns with the closing gradient EMA in
              genuine training. A splice decorrelates the two (the forger's
              delta points at the target, not along the claimed moments).

Forgery models (what a cheating publisher would actually do):

  * splice -- replace boundary k+1 with the same-architecture model trained
              identically from a different seed: right loss, right norms, the
              classic PoL spoof standing in for "trained on poisoned data".
  * interp -- move boundaries a fraction alpha toward a target model
              (the stealthy gradual splice; detection measured vs alpha).
  * edit   -- gaussian weight edit of relative size sigma (post-hoc tamper).

All detectors assume the LAZY forger who keeps the genuine optimizer moments
(forging bit-consistent moments for a fake trajectory without running the
committed training is the hard problem determinism creates). Honest scope:
these are heuristics that prioritize audits; the sound guarantee remains the
replay itself.

CPU smoke:
    python forgery.py --model mlp --segments 4 --segment-steps 3 \
        --batch-size 4 --block-size 48
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from artifacts import RUNS_DIR
from chain import CHAIN_MANIFEST_NAME, train_chain
from dataset import get_dataset
from model import build_model
from signing import verified_torch_load

DEFAULT_OUT = RUNS_DIR / "forgery_report.json"


# --------------------------------------------------------------------------- #
# Chain loading: pair each parameter with its Adam exp_avg, flattened
# --------------------------------------------------------------------------- #
def _flat_states(chain_dir, manifest):
    """Return per-boundary (weights, exp_avg) flat vectors, parameter-aligned.

    Optimizer state indices follow model.parameters() order (Adam was built
    from it), so the pairing is reconstructed by building the model from the
    committed config.
    """
    c = manifest["commitment"]
    dataset = get_dataset(c["dataset"], block_size=c["config"]["block_size"])
    model = build_model(c["model"], dataset.vocab_size, c["config"])
    param_names = [n for n, _ in model.named_parameters()]

    boundaries = []
    for b in manifest["boundaries"]:
        ckpt = verified_torch_load(Path(chain_dir) / b["file"], map_location="cpu")
        w = torch.cat([ckpt["model"][n].reshape(-1).float() for n in param_names])
        state = ckpt["optimizer"]["state"]
        if state:  # boundary 0 has no moments yet
            m = torch.cat([state[i]["exp_avg"].reshape(-1).float()
                           for i in range(len(param_names))])
        else:
            m = torch.zeros_like(w)
        boundaries.append({"w": w, "m": m})
    return boundaries


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def segment_scores(states, segment_steps, lr):
    """Per-segment replay-free statistics for a chain of flat states."""
    scores = []
    for k in range(len(states) - 1):
        dw = states[k + 1]["w"] - states[k]["w"]
        m = states[k + 1]["m"]
        scores.append({
            "segment": k,
            "reach": (dw.abs().max() / (segment_steps * lr)).item(),
            "norm": dw.norm().item(),
            "moment_cos": F.cosine_similarity(dw, -m, dim=0).item(),
        })
    norms = torch.tensor([s["norm"] for s in scores])
    mu, sd = norms.mean(), norms.std().clamp_min(1e-12)
    for s in scores:
        s["norm_z"] = ((s["norm"] - mu) / sd).item()
    return scores


def flag_segments(scores, thresholds):
    """Zero-false-positive flags: score beyond the genuine chain's envelope."""
    flags = []
    for s in scores:
        reasons = []
        if s["reach"] > thresholds["reach_max"]:
            reasons.append("reach")
        if abs(s["norm_z"]) > thresholds["norm_z_max"]:
            reasons.append("norm")
        if s["moment_cos"] < thresholds["moment_cos_min"]:
            reasons.append("moment")
        if reasons:
            flags.append({"segment": s["segment"], "reasons": reasons})
    return flags


def genuine_thresholds(scores, margin=1.25):
    """Detection envelope from the genuine chain, widened by a safety margin
    so a genuine chain never self-flags (zero false positives by construction)."""
    return {
        "reach_max": max(s["reach"] for s in scores) * margin,
        "norm_z_max": max(abs(s["norm_z"]) for s in scores) * margin,
        "moment_cos_min": min(s["moment_cos"] for s in scores)
        - (1 - min(s["moment_cos"] for s in scores)) * (margin - 1),
    }


# --------------------------------------------------------------------------- #
# Forgery constructors (operate on flat states; lazy forger keeps moments)
# --------------------------------------------------------------------------- #
def forge_splice(states, donor_states, k):
    """Boundary k+1 replaced by the alt-seed twin's boundary k+1."""
    forged = [dict(s) for s in states]
    forged[k + 1] = {"w": donor_states[k + 1]["w"], "m": states[k + 1]["m"]}
    return forged


def forge_interp(states, target_w, alpha):
    """Every boundary (except the committed init) moved alpha toward target."""
    forged = [dict(states[0])]
    for s in states[1:]:
        forged.append({"w": (1 - alpha) * s["w"] + alpha * target_w, "m": s["m"]})
    return forged


def forge_edit(states, k, sigma_rel, seed=0):
    """Gaussian edit of boundary k+1, sigma relative to the weight std."""
    forged = [dict(s) for s in states]
    g = torch.Generator().manual_seed(seed)
    w = states[k + 1]["w"]
    noise = torch.randn(w.shape, generator=g) * (w.std() * sigma_rel)
    forged[k + 1] = {"w": w + noise, "m": states[k + 1]["m"]}
    return forged


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #
def run_experiment(model_name="mlp", dataset_name="shakespeare", num_segments=4,
                   segment_steps=3, device="auto", overrides=None, alt_seed=7,
                   out_path=DEFAULT_OUT, chain_dir=None, donor_dir=None):
    """Train genuine + alt-seed chains, plant forgeries, score all detectors.

    Returns the report dict. report["all_detected"] is True iff every planted
    forgery was flagged; report["false_positives"] counts genuine flags (must
    be 0 by construction of the thresholds).
    """
    chain_dir = Path(chain_dir) if chain_dir else RUNS_DIR / "forgery_genuine"
    donor_dir = Path(donor_dir) if donor_dir else RUNS_DIR / "forgery_donor"

    t0 = time.time()
    manifest = train_chain(model_name, dataset_name, out_dir=chain_dir,
                           num_segments=num_segments, segment_steps=segment_steps,
                           device=device, overrides=overrides)
    donor_manifest = train_chain(model_name, dataset_name, out_dir=donor_dir,
                                 num_segments=num_segments,
                                 segment_steps=segment_steps, seed=alt_seed,
                                 device=device, overrides=overrides)

    states = _flat_states(chain_dir, manifest)
    donor = _flat_states(donor_dir, donor_manifest)
    cfg = manifest["commitment"]["config"]
    lr, steps = cfg["lr"], segment_steps

    genuine_scores = segment_scores(states, steps, lr)
    thresholds = genuine_thresholds(genuine_scores)
    fp = flag_segments(genuine_scores, thresholds)

    mid = num_segments // 2
    target_w = donor[-1]["w"]
    forgeries = {
        f"splice@{mid}": (forge_splice(states, donor, mid), [mid, mid + 1]),
        "interp@0.10": (forge_interp(states, target_w, 0.10), None),
        "interp@0.03": (forge_interp(states, target_w, 0.03), None),
        f"edit@{mid}_sigma1e-2": (forge_edit(states, mid, 1e-2), [mid, mid + 1]),
        f"edit@{mid}_sigma1e-4": (forge_edit(states, mid, 1e-4), [mid, mid + 1]),
    }
    # A splice/edit at boundary k+1 corrupts BOTH adjacent transitions (into and
    # out of the forged boundary); either flag counts as detection + localization.

    results = {}
    for name, (forged, affected) in forgeries.items():
        scores = segment_scores(forged, steps, lr)
        flags = flag_segments(scores, thresholds)
        flagged = [f["segment"] for f in flags]
        detected = bool(flags)
        localized = (affected is None) or bool(set(flagged) & set(affected))
        results[name] = {"detected": detected, "localized": detected and localized,
                         "flags": flags, "scores": scores}

    report = {
        "model": model_name,
        "dataset": dataset_name,
        "num_segments": num_segments,
        "segment_steps": steps,
        "num_params": manifest["num_params"],
        "device": manifest["environment"]["device_name"],
        "torch": manifest["environment"]["torch"],
        "genuine_scores": genuine_scores,
        "thresholds": thresholds,
        "false_positives": len(fp),
        "forgeries": results,
        "all_detected": all(r["detected"] for r in results.values()),
        "wall_time_s": round(time.time() - t0, 2),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[forgery] genuine chain: {len(genuine_scores)} segments, "
          f"false positives: {len(fp)}")
    for name, r in results.items():
        verdict = "DETECTED" + (" +localized" if r["localized"] else "") \
            if r["detected"] else "MISSED"
        via = ",".join(sorted({x for f in r['flags'] for x in f['reasons']})) or "-"
        print(f"  {name:<22} {verdict:<22} via {via}")
    print(f"[forgery] all_detected={report['all_detected']} -> {out_path}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mlp")
    p.add_argument("--dataset", default="shakespeare")
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--segment-steps", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--alt-seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    overrides = {}
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.block_size:
        overrides["block_size"] = args.block_size

    report = run_experiment(args.model, args.dataset, num_segments=args.segments,
                            segment_steps=args.segment_steps, device=args.device,
                            overrides=overrides, alt_seed=args.alt_seed,
                            out_path=args.out)
    sys.exit(0 if report["false_positives"] == 0 else 1)


if __name__ == "__main__":
    main()
