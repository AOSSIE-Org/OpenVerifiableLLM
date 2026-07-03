import os
import random

import numpy as np
import torch

# Importing device pins CUBLAS_WORKSPACE_CONFIG before the first CUDA op, which
# deterministic GPU matmuls require. Harmless (no-op) on CPU.
import device  # noqa: F401


def set_seed(seed: int = 99, deterministic: bool = True, warn_only: bool = False):
    """Seed Python/NumPy/torch + accelerators and configure determinism.

    ``deterministic`` defaults to True (the audit's control condition). The matrix
    runner passes ``deterministic=False`` for the "determinism OFF" column, and
    ``warn_only=True`` so a single op lacking a deterministic kernel warns instead
    of crashing an entire sweep.
    """
    # Belt-and-suspenders: also set the cuBLAS workspace here in case set_seed is
    # used before `device` is imported elsewhere. Read once at CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)          # python random module
    np.random.seed(seed)       # numpy
    torch.manual_seed(seed)    # torch CPU + accelerator host-side seed

    device.seed_accelerators(seed)  # explicitly seed every CUDA/XPU generator
    device.configure_determinism(enabled=deterministic, warn_only=warn_only)

    print(f"seed set to {seed} (deterministic={deterministic})")


if __name__ == "__main__":
    set_seed(99)

    x = torch.randn(3, 3)
    print("Tensor X:")
    print(x)
    print("Deterministic enabled:", torch.are_deterministic_algorithms_enabled())
