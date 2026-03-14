# run_experiment.py
#
# Broad pass to see what actually breaks reproducibility.
# Run each variation twice - if the hashes differ it's genuinely random,
# if they match but don't match baseline it's repeatable but different.
#
# num_workers > 0 is skipped on Windows - DataLoader spawns subprocesses
# that re-import this file and it deadlocks. See WHY.md.

import os
import platform
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import hash_file, set_seed


class TinyModel(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.fc(x))


def make_dataset():
    # in a function so Windows worker processes don't re-run this on import
    g = torch.Generator().manual_seed(99)
    return TensorDataset(
        torch.randn(50, 10, generator=g),
        torch.randn(50, 2, generator=g),
    )


def save_deterministic(state_dict, path):
    # write tensor bytes directly - no zip container, no timestamps
    with open(path, "wb") as f:
        for k, v in state_dict.items():
            f.write(k.encode())
            f.write(v.detach().cpu().contiguous().numpy().tobytes())


def train_and_save(
    path,
    dataset,
    base_seed=123,
    save_method="deterministic",
    unset_torch=False,
    unset_numpy=False,
    unset_python=False,
    shuffle=False,
    dropout=False,
    num_workers=0,
):
    set_seed(base_seed)

    if unset_torch:
        torch.seed()
    if unset_numpy:
        np.random.seed()
    if unset_python:
        random.seed()

    g = torch.Generator().manual_seed(base_seed) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
    )

    model = TinyModel(use_dropout=dropout)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()
    for _ in range(5):
        for x, y in loader:
            opt.zero_grad()
            nn.MSELoss()(model(x), y).backward()
            opt.step()

    if save_method == "torch":
        torch.save(model.state_dict(), path)
    else:
        save_deterministic(model.state_dict(), path)

    return hash_file(path)


def print_result(label, h1, h2, baseline):
    same = h1 == h2
    matches = h1 == baseline
    flag = "ok" if same else "DIFFERS"
    print(f"  [{flag}] {label}")
    print(f"         {h1[:12]}  {h2[:12]}  reproducible={same}  matches_baseline={matches}\n")


def run():
    dataset = make_dataset()
    files = []

    baseline = train_and_save("_base.pt", dataset)
    files.append("_base.pt")
    print(f"baseline:  {baseline[:12]}\n")

    # --- round 1: what breaks it ---
    print("--- things that break reproducibility ---\n")

    broken: List[Tuple[str, Dict[str, Any]]] = [
        ("torch rng not seeded", {"unset_torch": True}),
    ]

    if platform.system() != "Windows":
        broken.append(("num_workers=2 no seeded generator", {"num_workers": 2, "shuffle": True}))
    else:
        print("  [skipped] num_workers test - Windows only, see WHY.md\n")

    for label, kwargs in broken:
        path = f"_ckpt_{label[:20].replace(' ', '_')}.pt"
        h1 = train_and_save(path, dataset, **kwargs)
        h2 = train_and_save(path, dataset, **kwargs)
        files.append(path)
        print_result(label, h1, h2, baseline)

    # --- round 2: things that change output but are still deterministic ---
    print("--- things that change the outcome but stay deterministic ---\n")

    divergent: List[Tuple[str, Dict[str, Any]]] = [
        ("torch.save instead of det-save", {"save_method": "torch"}),
        ("shuffle same seed", {"shuffle": True}),
        ("dropout same seed", {"dropout": True}),
    ]

    for label, kwargs in divergent:
        path = f"_ckpt_{label[:20].replace(' ', '_')}.pt"
        h1 = train_and_save(path, dataset, **kwargs)
        h2 = train_and_save(path, dataset, **kwargs)
        files.append(path)
        print_result(label, h1, h2, baseline)

    # --- round 3: things that look like they should break it but don't ---
    print("--- things that turn out to be inert ---\n")

    inert: List[Tuple[str, Dict[str, Any]]] = [
        ("numpy rng not seeded + shuffle", {"unset_numpy": True, "shuffle": True}),
        ("python rng not seeded + shuffle", {"unset_python": True, "shuffle": True}),
    ]

    for label, kwargs in inert:
        path = f"_ckpt_{label[:20].replace(' ', '_')}.pt"
        h1 = train_and_save(path, dataset, **kwargs)
        h2 = train_and_save(path, dataset, **kwargs)
        files.append(path)
        print_result(label, h1, h2, baseline)

    for f in files:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    run()
