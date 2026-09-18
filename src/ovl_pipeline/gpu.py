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
REQUIRED_FLAGS = {
    "fp32_precision":"ieee", "matmul_fp32_precision":"ieee", "cudnn_fp32_precision":"ieee",
    "conv_fp32_precision":"ieee", "rnn_fp32_precision":"ieee", "bf16_reduced_precision":False,
    "bf16_split_k":False, "fp16_reduced_precision":False, "fp16_split_k":False,
    "fp16_accumulation":False, "cudnn_benchmark":False, "cudnn_deterministic":True,
    "deterministic":True, "warn_only":False, "fill_uninitialized_memory":True,
    "threads":1, "interop_threads":1,
}


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
    from .runtime_launch import current_launch
    current_launch()
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
    if flags() != REQUIRED_FLAGS:
        raise EvidenceError("GPU runtime did not achieve the declared numerical profile")
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
    if (not _configured or expected_flags != REQUIRED_FLAGS or flags() != REQUIRED_FLAGS or torch.cuda.current_device() != 0
            or next(model.parameters()).device != torch.device("cuda:0")):
        raise EvidenceError("GPU runtime changed after configuration")
    result=training.update(model, optimizer, batch, control, total,
                           precision=config["precision"], metrics=metrics)
    from .runtime_activity import update as observe_completed_update
    observe_completed_update(result)
    return result


def cpu_identity(raw):
    """Stable exposed x86 CPU descriptors; no hostname/serial/frequency fields."""
    if type(raw) is not str or len(raw.encode())>4*1024*1024:
        raise EvidenceError("CPU identity input oversized/invalid")
    descriptors=[]
    for block in raw.strip().split("\n\n"):
        values={}
        for line in block.splitlines():
            if ":" not in line:continue
            name,value=line.split(":",1);name=name.strip()
            if name in values:raise EvidenceError("duplicate CPU descriptor field")
            values[name]=value.strip()
        if "processor" not in values:continue
        required=("vendor_id","cpu family","model","model name","stepping","flags")
        if any(not values.get(k) for k in required):raise EvidenceError("incomplete x86 CPU identity")
        entry={k:values[k] for k in required if k!="flags"}
        entry["flags"]=sorted(set(values["flags"].split()))
        entry["microcode"]=values.get("microcode")
        if entry not in descriptors:descriptors.append(entry)
    if not descriptors:raise EvidenceError("missing x86 CPU descriptors")
    from .canonical import canonical
    return sorted(descriptors,key=canonical)


def host_runtime():
    """Record initialization-relevant host inputs, not an installed-stack audit."""
    if platform.system()!="Linux" or platform.machine()!="x86_64":
        raise EvidenceError("GPU host fingerprint supports Linux x86_64 only")
    declared=Path(sys.executable).resolve(strict=True)
    executable=Path("/proc/self/exe")
    actual_stat=executable.stat();declared_stat=declared.stat()
    if not declared.is_file() or (actual_stat.st_dev,actual_stat.st_ino)!=(declared_stat.st_dev,declared_stat.st_ino):
        raise EvidenceError("Python executable path differs from running interpreter")
    with Path("/proc/cpuinfo").open("r") as f:raw=f.read(4*1024*1024+1)
    return {"schema":"ovl.initialization-host.v1","cpu_descriptors":cpu_identity(raw),
            "torch_cpu_capability":torch.backends.cpu.get_cpu_capability(),
            "python_executable_sha256":file_hash(executable),"python_executable_bytes":executable.stat().st_size,
            "python_implementation":platform.python_implementation(),"python_build":list(platform.python_build()),
            "python_flags":str(sys.flags),"byteorder":sys.byteorder,
            "scope":"operator-observed CPU dispatch and interpreter bytes; not hardware attestation or full installed-stack verification"}


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
                or path.name.startswith(("libcuda.","libnvidia-","libpython")))):
            if not path.is_file():raise EvidenceError("mapped numerical library is missing/deleted")
            if str(path) not in libraries:
                libraries[str(path)] = {"name":path.name,"bytes":path.stat().st_size,"sha256":file_hash(path)}
    if not libraries or not any(e["name"].startswith("libcuda.") for e in libraries.values()):
        raise EvidenceError("missing mapped CUDA driver library identity")
    from .runtime_launch import current_launch
    compatible = {"schema":"ovl.gpu-environment.v1", "kernel":config,
        "installed_wheel_audit":current_launch(),
        "python":platform.python_version(),"machine":platform.machine(),"system":platform.system(),
        "initialization_host":host_runtime(),
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
