"""Device selection and accelerator determinism configuration (Phase 3).

This module centralizes everything that differs between the CPU baseline and an
accelerator run. It supports two accelerator backends:

* **CUDA** (NVIDIA GPUs) — deterministic cuDNN + a pinned cuBLAS workspace.
* **XPU**  (Intel GPUs, e.g. Iris Xe / Arc, via the oneAPI backend) — determinism
  rides on ``torch.use_deterministic_algorithms(True)``; there is no cuBLAS-style
  workspace knob, so bitwise reproducibility on XPU is best-effort.

Importing this module has one important side effect: it pins the cuBLAS workspace
*before* the first CUDA op, which is a hard requirement for deterministic matmuls
on CUDA >= 10.2. The env var is ignored by the XPU/CPU backends, so it is safe to
set unconditionally.
"""

import os
import warnings

# cuBLAS chooses its GEMM (matmul) reduction order based on a workspace it
# allocates lazily on the first CUDA call. A fixed workspace forces a single,
# reproducible reduction order. Read once at CUDA context creation, so it MUST be
# set before any tensor touches a CUDA device. Harmless on XPU/CPU.
# https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def _xpu_available():
    """True when a usable Intel XPU backend is present."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def accelerator_module():
    """Return the active accelerator's namespace (``torch.cuda`` or ``torch.xpu``).

    CUDA is preferred when both are present; ``None`` means CPU-only. Every
    accelerator-specific call in this module goes through here, so adding a new
    backend is a one-line change.
    """
    if torch.cuda.is_available():
        return torch.cuda
    if _xpu_available():
        return torch.xpu
    return None


def get_device():
    """Return the best available device: CUDA, else Intel XPU, else CPU.

    The same code path runs on all three; only the floating-point reduction order
    (the hardware entropy under study) differs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _xpu_available():
        return torch.device("xpu")
    return torch.device("cpu")


def device_name(device=None):
    """Human-readable name for a device (for fingerprints/logs)."""
    device = device or get_device()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "xpu":
        return torch.xpu.get_device_name(device)
    return "cpu"


def seed_accelerators(seed):
    """Seed every generator on the active accelerator (CUDA or XPU). No-op on CPU."""
    accel = accelerator_module()
    if accel is not None:
        accel.manual_seed_all(seed)


def accel_rng_state():
    """Accelerator RNG state tagged with its backend, or ``None`` on CPU.

    Dropout (kept active at 0.1) draws from the accelerator's generator on GPU, so
    this state must be serialized alongside the CPU/NumPy/Python RNG for a
    segmented replay to stay deterministic. The backend tag lets a resume safely
    skip state that was captured on a different backend (e.g. CUDA vs XPU).
    """
    accel = accelerator_module()
    if accel is None:
        return None
    return {"backend": get_device().type, "state": accel.get_rng_state_all()}


def restore_accel_rng_state(saved):
    """Restore state from :func:`accel_rng_state`. No-op on CPU or on mismatch."""
    if not saved:
        return
    accel = accelerator_module()
    if accel is None:
        return
    current = get_device().type
    if saved.get("backend") != current:
        warnings.warn(
            f"Skipping accelerator RNG restore: checkpoint backend "
            f"{saved.get('backend')!r} != current backend {current!r}."
        )
        return
    accel.set_rng_state_all(saved["state"])


def configure_determinism():
    """Apply backend-appropriate determinism settings.

    ``torch.use_deterministic_algorithms(True)`` is backend-agnostic; the cuDNN
    flags only matter on CUDA, and there is no XPU equivalent.
    """
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
