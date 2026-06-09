import hashlib
import json
import platform
import subprocess
import sys
from typing import Any, Dict


def _canonical_json(obj: Any) -> str:
    """
    Serialize object into canonical JSON format.
    Ensures stable hashing across runs.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_object_hash(obj: Any) -> str:
    """
    Compute SHA-256 hash of a JSON-serializable object.
    Uses canonical JSON serialization to ensure stable, order-independent hashing.
    """
    canonical = _canonical_json(obj)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_environment_metadata() -> Dict[str, Any]:
    """
    Collect runtime environment metadata relevant to training.
    """
    env: Dict[str, Any] = {}

    # Python + OS
    env["python_version"] = sys.version
    env["platform"] = platform.platform()

    # PyTorch + CUDA
    try:
        import torch

        env["pytorch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        try:
            env["cudnn_version"] = torch.backends.cudnn.version()
        except (AttributeError, ImportError, RuntimeError):
            env["cudnn_version"] = None

        # GPU info
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_properties"] = {
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "compute_capability_major": props.major,
                "compute_capability_minor": props.minor,
            }
        else:
            env["gpu_name"] = None
            env["gpu_properties"] = None
    except ImportError:
        env["pytorch_version"] = None
        env["cuda_version"] = None
        env["cudnn_version"] = None
        env["gpu_name"] = None
        env["gpu_properties"] = None

    # NVIDIA driver
    try:
        driver_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        drivers = sorted(
            list(set(line.strip() for line in driver_output.splitlines() if line.strip()))
        )
        env["nvidia_driver"] = ", ".join(drivers)
    except Exception:
        env["nvidia_driver"] = None

    # Installed packages
    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()

        packages.sort()
        env["pip_packages"] = packages
    except Exception:
        env["pip_packages"] = []

    return env


def generate_environment_fingerprint() -> Dict[str, Any]:
    """
    Generate environment metadata + environment_hash.
    Returns a dictionary that can be embedded into existing manifest.
    """
    metadata = collect_environment_metadata()
    environment_hash = compute_object_hash(metadata)

    return {
        "environment": metadata,
        "environment_hash": environment_hash,
    }
