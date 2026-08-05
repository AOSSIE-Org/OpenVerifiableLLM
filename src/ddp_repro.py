#!/usr/bin/env python3
"""T9 (STRETCH) -- does DDP all-reduce preserve bitwise reproducibility?

    torchrun --nproc_per_node=2 ddp_repro.py        # from src/, on a >=2-GPU pod

Each rank trains the same gpt10m on the same data; DDP averages gradients via
all-reduce. NCCL all-reduce is not guaranteed to reduce in a fixed order, so even
with deterministic per-rank kernels the averaged gradient -- and hence the model --
can differ run to run. We train twice and compare the final parameter hash, and we
compare hashes across ranks.

A NEGATIVE result (cannot get bitwise-identical run-to-run under DDP) is the honest,
interesting outcome -- it is the open problem, not a bug to hide. Keep this last; do
not let it become the main thrust.

Distributed Determinism Roadmap & Mitigation Strategies:
---------------------------------------------------------
1. FP64 Gradient Accumulation:
   Accumulating gradients in FP64 before applying optimizer steps reduces the
   nondeterministic precision discrepancies introduced by varying floating-point
   reduction orders in NCCL all-reduce operations.
2. Deterministic Communication Backends / Wrappers:
   Forcing a deterministic reduction tree (such as sorting elements or forcing
   a single reduction order) ensures that averaged gradients are mathematically
   and bitwise identical across all parallel nodes/ranks, at the cost of some
   inter-GPU communication overhead.
"""
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import model_config
from dataset import get_dataset
from main import set_seed
from model import build_model


def hash_model(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_once(local_rank, cfg):
    set_seed(cfg["seed"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    ds = get_dataset("shakespeare", block_size=cfg["block_size"])
    model = build_model("gpt10m", ds.vocab_size, cfg).to(dev)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    for _ in range(cfg["total_steps"]):
        x, y = ds.get_batch(cfg["batch_size"], cfg["block_size"], device=dev)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return hash_model(model.module)


def main():
    if "RANK" not in os.environ:
        print("Launch with: torchrun --nproc_per_node=2 ddp_repro.py")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg = model_config("gpt10m", total_steps=20)

    h1 = train_once(local_rank, cfg)
    h2 = train_once(local_rank, cfg)
    run_to_run_same = h1 == h2

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, h1)
    cross_rank_same = len(set(gathered)) == 1

    if rank == 0:
        print(f"[DDP] run-to-run bitwise identical: {run_to_run_same}")
        print(f"[DDP] all ranks identical:          {cross_rank_same}")
        print(f"[DDP] rank hashes: {[h[:12] for h in gathered]}")
        if not run_to_run_same:
            print("[DDP] NCCL all-reduce ordering breaks bitwise reproducibility even with "
                  "deterministic per-rank kernels -- this is the open problem, present it as one.")
            print("\n[DDP] Distributed Determinism Roadmap mitigations to consider:")
            print("  1. FP64 Gradient Accumulation: Accumulating gradients in FP64 reduces order-of-operation precision discrepancies.")
            print("  2. Deterministic Communication Wrappers: Force a deterministic reduction order across nodes/ranks.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
