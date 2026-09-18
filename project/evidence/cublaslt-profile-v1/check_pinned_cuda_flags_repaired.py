"""Host-only real CUDA-wheel flag check. Does not initialize or test any GPU."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/'src'),str(root/'.ovllm-cache/offline-runtime-setup-v2/runtime/venv/lib/python3.12/site-packages')]
import torch
from ovl_pipeline import gpu
from ovl_pipeline.canonical import digest,file_hash,write_json
assert torch.__version__=='2.14.0+cu130' and torch.version.cuda=='13.0'
assert not torch.cuda.is_initialized()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
negative_control={}
for dtype in ('fp16','bf16'):
    torch.backends.cuda.preferred_blas_library('cublas')
    setattr(torch.backends.cuda.matmul,'allow_'+dtype+'_reduced_precision_reduction',(False,False))
    negative_control[dtype]='setter accepted; no GPU GEMM executed'
gpu._set_flags();observed=gpu.flags();assert observed==gpu.REQUIRED_FLAGS
# Deliberate backend drift must differ despite unchanged precision flags.
torch.backends.cuda.preferred_blas_library('cublas');assert gpu.flags()!=gpu.REQUIRED_FLAGS
assert not torch.cuda.is_initialized()
write_json(Path(sys.argv[1]),{'schema':'ovl.host-cuda-flags-check.v1','result':'PASS','torch':str(torch.__version__),'cuda_build':torch.version.cuda,'gpu_py_sha256':file_hash(root/'src/ovl_pipeline/gpu.py'),'observed_flags':observed,'backend_drift_detected':True,'cuda_initialized':False,'negative_control':negative_control,'workspace_validation':'NOT_RUN_REQUIRES_GPU_DRIVER','scope':'real pinned CUDA wheel setters/readback on local CPU host only; actual CUDA arithmetic and replay NOT_RUN'})
