#!/usr/bin/env python3
"""T7 -- divergence-accumulation plot from a --track-divergence record.

Reads a results .jsonl whose records carry per-step parameter hashes for two twin
runs (per_step_param_sha256_runA / _runB) and plots, per cell, the cumulative count
of steps at which the two "identical" runs have diverged, marking the first one.

    python src/plot_divergence.py results/divergence.jsonl   # -> results/divergence.png

Torch-free (json + matplotlib only).
"""
import json
import sys
from pathlib import Path


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def divergence_signal(a, b):
    n = min(len(a), len(b))
    return [0 if a[i] == b[i] else 1 for i in range(n)]


def first_divergence(signal):
    for i, v in enumerate(signal):
        if v:
            return i
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/divergence.jsonl"
    recs = [r for r in load_records(path)
            if r.get("per_step_param_sha256_runA") and r.get("per_step_param_sha256_runB")]
    if not recs:
        print("No per-step divergence data found. Re-run with --track-divergence, e.g.:\n"
              "  python run_experiment.py --model gpt10m --precision fp32 --deterministic off "
              "--device cuda --track-divergence --out results/divergence.jsonl")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in recs:
        sig = divergence_signal(r["per_step_param_sha256_runA"], r["per_step_param_sha256_runB"])
        cum, s = [], 0
        for v in sig:
            s += v
            cum.append(s)
        label = f"{r['model']}/{r.get('condition', r['precision'])} ({r['device']})"
        ax.plot(range(len(cum)), cum, marker=".", label=label)
        fd = first_divergence(sig)
        if fd is not None:
            ax.axvline(fd, ls="--", alpha=0.4)
            ax.annotate(f"first divergence @ step {fd}", (fd, 0.2),
                        rotation=90, va="bottom", fontsize=8)

    ax.set_xlabel("training step")
    ax.set_ylabel("cumulative diverged steps")
    ax.set_title("Divergence accumulation between two 'identical' runs (T7)")
    ax.legend(fontsize=8)
    out = Path(path).with_name("divergence.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for r in recs:
        sig = divergence_signal(r["per_step_param_sha256_runA"], r["per_step_param_sha256_runB"])
        print(f"  {r['model']}/{r.get('condition', r['precision'])}: "
              f"first divergence at step {first_divergence(sig)}")


if __name__ == "__main__":
    main()
