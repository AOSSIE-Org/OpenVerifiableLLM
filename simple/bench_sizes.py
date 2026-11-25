"""Size the next model: time the exact deterministic training step for several TinyGPT sizes
on the real Wikipedia stream, and project one full pass (train + checking every segment).
    python simple/bench_sizes.py --data PREP [--seconds 120] [--price 1.19]
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ovl  # noqa: E402

SIZES = [(384, 6), (512, 8), (640, 10), (768, 12), (1024, 16)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--price", type=float, default=1.19)
    a = p.parse_args()
    updates = ovl.RECIPE["schedule"]["wikipedia_updates"]
    for dim, layers in SIZES:
        ovl.RECIPE["model"].update(embed_dim=dim, num_layers=layers, num_heads=dim // 64)
        try:
            model, opt = ovl.fresh()
        except RuntimeError as e:
            print(json.dumps({"embed_dim": dim, "layers": layers, "error": str(e)[:200]}), flush=True)
            continue
        params = sum(t.numel() for t in model.parameters())
        ctl, n, t0, elapsed = ovl.initial_control(), 0, None, None
        with contextlib.closing(ovl.prefetch(a.data, "wikipedia", [0, 0])) as batches:
            for b, _ in batches:
                _, ctl["cursor"], _ = ovl.step(model, opt, b, ctl)
                n += 1
                if n == 30:
                    if ovl.DEV == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                elif t0 and time.time() - t0 >= a.seconds:
                    break
            if ovl.DEV == "cuda":
                torch.cuda.synchronize()
            if t0 is not None:
                elapsed = time.time() - t0
        if elapsed is None or n <= 30:
            sys.exit(f"stream ended after {n} updates (need > 30 to benchmark)")
        ms = 1000 * elapsed / (n - 30)
        hours = updates * ms / 3.6e6
        ckpt_gb = params * 12 / 1e9  # fp32 weights + two Adam moments
        peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 1) if ovl.DEV == "cuda" else 0.0
        print(json.dumps({"embed_dim": dim, "layers": layers, "params_m": round(params / 1e6, 1),
                          "ms_per_update": round(ms, 1), "train_hours": round(hours, 1),
                          "train_plus_full_check_usd": round(2 * hours * a.price, 1),
                          "checkpoint_gb": round(ckpt_gb, 2), "all_checkpoints_gb": round(ckpt_gb * 32, 0),
                          "peak_mem_gb": peak_gb}), flush=True)
        del model, opt
        if ovl.DEV == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
