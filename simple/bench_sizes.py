"""Size the next model: time the exact deterministic training step for several TinyGPT sizes
on the real Wikipedia stream, and project one full pass (train + checking every segment).
    python simple/bench_sizes.py --data PREP [--seconds 120] [--price 1.19]
"""
import argparse
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
        ctl, n, t0 = ovl.initial_control(), 0, None
        for b, _ in ovl.prefetch(a.data, "wikipedia", [0, 0]):
            _, ctl["cursor"], _ = ovl.step(model, opt, b, ctl)
            n += 1
            if n == 30:
                torch.cuda.synchronize(); t0 = time.time()
            if t0 and time.time() - t0 >= a.seconds:
                break
        torch.cuda.synchronize()
        ms = 1000 * (time.time() - t0) / (n - 30)
        hours = updates * ms / 3.6e6
        ckpt_gb = params * 12 / 1e9  # fp32 weights + two Adam moments
        print(json.dumps({"embed_dim": dim, "layers": layers, "params_m": round(params / 1e6, 1),
                          "ms_per_update": round(ms, 1), "train_hours": round(hours, 1),
                          "train_plus_full_check_usd": round(2 * hours * a.price, 1),
                          "checkpoint_gb": round(ckpt_gb, 2), "all_checkpoints_gb": round(ckpt_gb * 32, 0),
                          "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}), flush=True)
        del model, opt
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
