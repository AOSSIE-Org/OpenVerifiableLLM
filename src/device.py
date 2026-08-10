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
    # State tensors may have been moved to the accelerator by a checkpoint's
    # map_location; set_rng_state_all requires CPU ByteTensors.
    state = saved["state"]
    if isinstance(state, (list, tuple)):
        state = [s.cpu() for s in state]
    accel.set_rng_state_all(state)


def configure_determinism(enabled=True, warn_only=False):
    """Apply (or deliberately remove) backend-appropriate determinism settings.

    ``enabled=True`` is the control condition. ``enabled=False`` is the
    "determinism OFF" matrix column: it lets the framework pick fast,
    nondeterministic kernels (notably atomicAdd in embedding/scatter backward and
    cuDNN benchmark autotuning) so we can MEASURE run-to-run divergence rather
    than assume it.

    ``warn_only=True`` (used by the sweep) downgrades "no deterministic kernel for
    this op" from a hard error to a warning, so a single exotic op can't crash a
    whole matrix run. The audit's control cell uses ``warn_only=False`` for the
    strongest possible bitwise claim.
    """
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


# --------------------------------------------------------------------------- #
# Precision control (fp32 / tf32 / bf16 / fp16)
# --------------------------------------------------------------------------- #
from contextlib import nullcontext  # noqa: E402

VALID_PRECISIONS = ("fp32", "tf32", "bf16", "fp16")


def apply_precision(mode):
    """Configure the matmul backend for a precision mode. Returns the mode.

    * ``fp32``  - true IEEE single precision; TF32 explicitly OFF (the honest
                  reference).
    * ``tf32``  - allow TF32 tensor-core matmuls. On Ampere+ GPUs this is the
                  *default*, which is exactly why it is the "silent killer":
                  results stop matching the fp32 reference without any visible
                  code change. On CPU/pre-Ampere this is a no-op (so a tf32 cell
                  there will read as PASS -- TF32 divergence requires the hardware).
    * ``bf16`` / ``fp16`` - handled at the op level via :func:`autocast_context`;
                  here we just leave TF32 off so the only precision change is the
                  autocast itself.
    """
    mode = (mode or "fp32").lower()
    if mode not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {VALID_PRECISIONS}, got {mode!r}")

    use_tf32 = mode == "tf32"
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = use_tf32
    # Newer torch funnels the same knob through this API.
    try:
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    except Exception:
        pass
    return mode


def autocast_context(mode, device_type=None):
    """Return the autocast context manager for a precision mode (else nullcontext).

    bf16 autocast works on both CPU and CUDA; fp16 autocast is GPU-oriented.
    """
    mode = (mode or "fp32").lower()
    if mode in ("bf16", "fp16"):
        dtype = torch.bfloat16 if mode == "bf16" else torch.float16
        device_type = device_type or get_device().type
        # Guard fp16 autocast on CPU
        if mode == "fp16" and device_type == "cpu":
            raise ValueError(
                "fp16 autocast is not supported on CPU. Use bf16 for reduced precision on CPU, "
                "or run on a CUDA/XPU device for fp16."
            )
        return torch.autocast(device_type=device_type, dtype=dtype)
    return nullcontext()


def precision_flags():
    """Snapshot of the current precision-related backend flags (for manifests)."""
    flags = {}
    try:
        flags["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        flags["cuda_matmul_allow_tf32"] = None
    try:
        flags["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        flags["cudnn_allow_tf32"] = None
    try:
        flags["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        flags["float32_matmul_precision"] = None
    flags["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    return flags


# precision/determinism knobs consumed by experiment.prepare_run and main.set_seed



def sdpa_context(backend=None):
    """Context selecting the scaled-dot-product-attention backend by name
    ("math" / "flash" / "efficient"); no-op when backend is falsy."""
    if not backend:
        from contextlib import nullcontext
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    mapping = {"math": SDPBackend.MATH,
               "flash": SDPBackend.FLASH_ATTENTION,
               "efficient": SDPBackend.EFFICIENT_ATTENTION}
    return sdpa_kernel([mapping[backend]])
