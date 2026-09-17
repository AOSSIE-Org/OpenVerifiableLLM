"""Explicit single-CUDA-device runtime for the shared update/state kernel.

Configuration is not a reproducibility result or production admission. Actual
fresh-process/resume pilots and a separately anchored registration are required.
"""
from __future__ import annotations

import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from .canonical import EvidenceError, file_hash
from .schema import fields
from . import training

WORKSPACE = ":4096:8"
_configured = False


def validate_config(value):
    fields(value, "schema precision", "GPU kernel configuration")
    if value["schema"] != "ovl.gpu-kernel.v1" or value["precision"] not in ("bf16", "fp32"):
        raise EvidenceError("unsupported GPU kernel configuration")


def _set_flags():
    # PyTorch 2.14's new precision API; do not mix it with legacy allow_tf32.
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True


def flags():
    return {"fp32_precision": torch.backends.fp32_precision,
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision,
            "conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "rnn_fp32_precision": torch.backends.cudnn.rnn.fp32_precision,
            "bf16_reduced_precision": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "bf16_split_k": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
            "fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "fp16_split_k": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k,
            "fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}


def configure(config):
    global _configured
    validate_config(config)
    if not torch.__version__.startswith("2.14.0+") or torch.version.cuda is None:
        raise EvidenceError("GPU profile requires an explicitly pinned PyTorch 2.14.0 CUDA build")
    required = {"CUBLAS_WORKSPACE_CONFIG":WORKSPACE, "TOKENIZERS_PARALLELISM":"false", "CUDA_VISIBLE_DEVICES":"0",
                "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
                "PYTHONHASHSEED":"0", "USE_PYTORCH_KERNEL_CACHE":"0"}
    if any(os.environ.get(k) != v for k,v in required.items()):
        raise EvidenceError("GPU process must start with the declared deterministic environment")
    if sys.flags.hash_randomization != 0:
        raise EvidenceError("PYTHONHASHSEED=0 must be set before interpreter startup")
    if any(os.environ.get(k) not in (None,"0") for k in ("NVIDIA_TF32_OVERRIDE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")):
        raise EvidenceError("TF32 environment override conflicts with IEEE profile")
    if torch.cuda.is_initialized() and not _configured:
        raise EvidenceError("CUDA initialized before explicit kernel configuration; start a fresh process")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EvidenceError("exactly one visible CUDA GPU required; no CPU fallback")
    if not _configured:
        torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    _set_flags()
    torch.cuda.set_device(0)
    torch.cuda.init()
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise EvidenceError("native GPU BF16 support required")
    _configured = True
    return flags()


def initialize(recipe, config):
    configure(config)
    model, optimizer, control = training.initialize(recipe, device="cuda:0")
    torch.cuda.synchronize()
    return model, optimizer, control


def update(model, optimizer, batch, control, total, config, *, expected_flags, metrics=None):
    validate_config(config)
    if (not _configured or flags() != expected_flags or torch.cuda.current_device() != 0
            or next(model.parameters()).device != torch.device("cuda:0")):
        raise EvidenceError("GPU runtime changed after configuration")
    return training.update(model, optimizer, batch, control, total,
                           precision=config["precision"], metrics=metrics)


def environment(config):
    """Compatibility fingerprint plus separately reported physical-device identity.

    Loaded library bytes and distribution RECORD identities supplement the required
    immutable container and wheel lock. This function alone cannot attest the host
    or freeze dependencies. Capture after the declared warmup loads numerical libs.
    """
    validate_config(config)
    if not _configured:
        raise EvidenceError("GPU runtime has not been configured")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                     "--format=csv,noheader,nounits"], text=True)
    devices = list(csv.reader(query.splitlines(), skipinitialspace=True))
    if len(devices) != 1 or len(devices[0]) != 4:
        raise EvidenceError("provider GPU inventory is not one physical GPU")
    _, _, gpu_uuid, driver = devices[0]
    p = torch.cuda.get_device_properties(0)
    cudnn_version = torch.backends.cudnn.version()  # Load before mapped-library capture.
    distributions = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name in {"torch","numpy","safetensors","pynacl","triton"} or name.startswith("nvidia-"):
            record = next((f for f in (dist.files or []) if str(f).endswith(".dist-info/RECORD")),None)
            if record is None:
                raise EvidenceError("numerical dependency lacks wheel RECORD")
            distributions.append({"name":name,"version":dist.version,"record_sha256":file_hash(Path(dist.locate_file(record)))})
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[-1].startswith("/"):
            continue
        path = Path(parts[-1])
        if (".so" in path.name and ("/torch/" in str(path) or "/nvidia/" in str(path)
                or path.name.startswith(("libcuda.","libnvidia-")))):
            if not path.is_file():raise EvidenceError("mapped numerical library is missing/deleted")
            if str(path) not in libraries:
                libraries[str(path)] = {"name":path.name,"bytes":path.stat().st_size,"sha256":file_hash(path)}
    if not libraries or not any(e["name"].startswith("libcuda.") for e in libraries.values()):
        raise EvidenceError("missing mapped CUDA driver library identity")
    compatible = {"schema":"ovl.gpu-environment.v1", "kernel":config,
        "python":platform.python_version(),"machine":platform.machine(),"system":platform.system(),
        "torch_build":torch.__config__.show(),"cuda_build":torch.version.cuda,
        "cudnn_version":cudnn_version,"driver_version":driver,
        "gpu":{"name":p.name,"compute_capability":[p.major,p.minor],"memory_bytes":p.total_memory,
               "multiprocessors":p.multi_processor_count,"warp_size":getattr(p,"warp_size",None)},
        "flags":flags(),"packages":sorted(distributions,key=lambda e:e["name"]),
        "loaded_numerical_libraries":sorted(libraries.values(),key=lambda e:(e["name"],e["sha256"])),
        "environment":{n:os.environ.get(n) for n in ["CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES",
            "CUDA_LAUNCH_BLOCKING","CUDA_MODULE_LOADING","CUDA_CACHE_DISABLE","NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF",
            "USE_PYTORCH_KERNEL_CACHE","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
            "PYTHONHASHSEED","TOKENIZERS_PARALLELISM"]},
        "attention":"manual","compiler":"eager-no-compile","grad_scaler":"none",
        "autocast_cache":False,"parameter_dtype":"float32","optimizer":"explicit-nonfused-AdamW-v1"}
    return {"compatible":compatible,"physical_observation":{"gpu_uuid":gpu_uuid,"nvidia_smi":query},
            "reproducibility_validation":"NOT_RUN","production_admission":"NOT_RUN"}
