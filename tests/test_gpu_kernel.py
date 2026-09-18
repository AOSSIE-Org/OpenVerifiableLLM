"""CPU checks of shared-kernel behavior; no claim of measured CUDA exactness."""
import math
import struct
import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from test_pipeline import prepared
from ovl_pipeline import gpu, schema
from ovl_pipeline.canonical import EvidenceError, digest
from ovl_pipeline.data import batches, check_coverage
from ovl_pipeline.fixture import recipe
from ovl_pipeline.state import capture, state_root, tensor_digest
from ovl_pipeline.training import initialize, update


def reference_update(model, optimizer, batch, control, total):
    """Pre-extension CPU arithmetic, to detect unintended training changes."""
    cursor=check_coverage(batch,control["cursor"],total)
    logits=model(batch["inputs"])
    losses=F.cross_entropy(logits.flatten(0,1),batch["targets"].flatten(),reduction="none")
    valid=batch["mask"].flatten()
    loss=losses[valid].sum()/valid.sum()
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    return {**control,"global_step":control["global_step"]+1,"phase_step":control["phase_step"]+1,
            "cursor":cursor,"transcript":digest({"previous":control["transcript"],"batch":tensor_digest(batch)})}


def test_shared_update_and_telemetry_preserve_cpu_trajectory(prepared):
    root,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"]);results=[]
    for mode in ("reference","shared","metrics"):
        model,opt,control=initialize(r)
        for batch in batches(root/"wikipedia",r["context"],r["batch_size"]):
            total=manifest["streams"]["wikipedia"]["targets"]
            if mode=="reference":control=reference_update(model,opt,batch,control,total)
            else:
                metrics={} if mode=="metrics" else None
                control=update(model,opt,batch,control,total,metrics=metrics)
                if metrics is not None:
                    assert math.isfinite(struct.unpack(">d",bytes.fromhex(metrics["loss_float64_hex"]))[0])
                    assert metrics["targets"]==int(batch["mask"].sum())
        results.append(state_root(*capture(model,opt,control)))
    assert results[0]==results[1]==results[2]


@pytest.mark.parametrize("precision",["bf16","fp16","tf32","automatic"])
def test_no_unsupported_precision_or_cpu_bf16_fallback(prepared,precision):
    root,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"])
    model,opt,control=initialize(r);before=state_root(*capture(model,opt,control))
    batch=next(batches(root/"wikipedia",r["context"],r["batch_size"]))
    with pytest.raises(EvidenceError):update(model,opt,batch,control,manifest["streams"]["wikipedia"]["targets"],precision=precision)
    assert state_root(*capture(model,opt,control))==before


def test_non_fp32_master_parameters_rejected(prepared):
    root,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"])
    model,opt,control=initialize(r);model.double()
    batch=next(batches(root/"wikipedia",r["context"],r["batch_size"]))
    with pytest.raises(EvidenceError,match="FP32 master"):
        update(model,opt,batch,control,manifest["streams"]["wikipedia"]["targets"])


def test_candidate_gpu_recipe_has_explicit_bounded_allocation_profile():
    r=recipe(32000);r.update(context=512,batch_size=8)
    r["model"].update(embed_dim=384,num_heads=6,num_layers=6,max_seq_len=512)
    with pytest.raises(EvidenceError,match="allocation"):schema.recipe(r)
    schema.recipe(r,gpu=True)
    r["batch_size"]=128
    with pytest.raises(EvidenceError,match="allocation"):schema.recipe(r,gpu=True)


def test_gpu_config_is_closed_and_requires_cuda_build(monkeypatch):
    value={"schema":"ovl.gpu-kernel.v1","precision":"bf16"};gpu.validate_config(value)
    for change in ({"precision":"fp16"},{"schema":"future"},{"cpu_fallback":True}):
        with pytest.raises(EvidenceError):gpu.validate_config({**value,**change})
    monkeypatch.setattr(torch.version,"cuda",None)
    with pytest.raises(EvidenceError,match="CUDA build"):gpu.configure(value)
    assert gpu._configured is False


def test_gpu_environment_cannot_be_reported_without_actual_initialization():
    with pytest.raises(EvidenceError,match="not been configured"):
        gpu.environment({"schema":"ovl.gpu-kernel.v1","precision":"fp32"})


def test_cuda_initialization_cannot_silently_use_unconfigured_device(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    with pytest.raises(EvidenceError,match="explicitly configured"):
        initialize(recipe(320),device="cuda:0")
    with pytest.raises(EvidenceError,match="unsupported kernel device"):
        initialize(recipe(320),device="cuda:1")


@pytest.mark.parametrize("noop", [False, True])
def test_precision_setters_against_literal_profile_and_noop_rejected(noop):
    # Execute actual precision setters in a fresh CPU process; fake only CUDA
    # availability/initialization and audited-launch admission. This is isolated
    # precision configuration coverage, not GPU or installed-runtime proof.
    code = '''
import torch
from ovl_pipeline import gpu
from ovl_pipeline import runtime_launch
from ovl_pipeline.canonical import EvidenceError
runtime_launch.current_launch=lambda:{'interpreter_origin':{'explicit-test-double':True}}  # flags-only launcher double
torch.__version__="2.14.0+cu130";torch.version.cuda="13.0"
torch.cuda.is_initialized=lambda:False;torch.cuda.is_available=lambda:True
torch.cuda.device_count=lambda:1;torch.cuda.set_device=lambda n:None
torch.cuda.init=lambda:None;torch.cuda.is_bf16_supported=lambda **k:True
noop=NOOP
if noop:
    torch.backends.cuda.matmul.fp32_precision="tf32"
    gpu._set_flags=lambda:None
    try:gpu.configure({"schema":"ovl.gpu-kernel.v1","precision":"bf16"})
    except EvidenceError as e:assert "declared numerical profile" in str(e)
    else:raise AssertionError("no-op profile was accepted")
else:
    got=gpu.configure({"schema":"ovl.gpu-kernel.v1","precision":"bf16"})
    assert got=={
      "fp32_precision":"ieee","matmul_fp32_precision":"ieee","cudnn_fp32_precision":"ieee",
      "conv_fp32_precision":"ieee","rnn_fp32_precision":"ieee","bf16_reduced_precision":False,
      "bf16_split_k":False,"fp16_reduced_precision":False,"fp16_split_k":False,
      "fp16_accumulation":False,"cudnn_benchmark":False,"cudnn_deterministic":True,
      "deterministic":True,"warn_only":False,"fill_uninitialized_memory":True,
      "threads":1,"interop_threads":1}
'''.replace("NOOP", repr(noop))
    env={**os.environ,"PYTHONHASHSEED":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8","CUDA_VISIBLE_DEVICES":"0",
         "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
         "USE_PYTORCH_KERNEL_CACHE":"0","NVIDIA_TF32_OVERRIDE":"0","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE":"0"}
    subprocess.run([sys.executable,"-c",code],env=env,check=True,capture_output=True,text=True)


def test_host_fingerprint_observes_actual_interpreter_and_cpu_without_claiming_cuda():
    from pathlib import Path
    from ovl_pipeline.canonical import file_hash
    a=gpu.host_runtime();b=gpu.host_runtime()
    assert a==b and a['python_executable_sha256']==file_hash(Path(sys.executable).resolve())
    assert a['cpu_descriptors'] and a['torch_cpu_capability']==torch.backends.cpu.get_cpu_capability()
    assert 'not hardware attestation' in a['scope']
    assert gpu._configured is False


def test_cpu_identity_ignores_order_and_clock_but_binds_dispatch_inputs():
    raw='processor : 0\nvendor_id : Vendor\ncpu family : 6\nmodel : 1\nmodel name : CPU\nstepping : 1\nflags : avx2 sse avx2\nmicrocode : 0x1\ncpu MHz : 2000\n'
    first=gpu.cpu_identity(raw)
    assert first==gpu.cpu_identity(raw.replace('2000','3000').replace('avx2 sse avx2','sse avx2')+'\n'+raw.replace('processor : 0','processor : 1'))
    for old,new in [('model : 1','model : 2'),('stepping : 1','stepping : 2'),('avx2','avx512'),('0x1','0x2')]:
        assert gpu.cpu_identity(raw.replace(old,new))!=first
    with pytest.raises(EvidenceError):gpu.cpu_identity(raw.replace('flags : avx2 sse avx2\n',''))
    with pytest.raises(EvidenceError):gpu.cpu_identity('')
    with pytest.raises(EvidenceError):gpu.cpu_identity(raw+'flags : other\n')


def test_host_fingerprint_rejects_replaced_interpreter_path(tmp_path,monkeypatch):
    other=tmp_path/'python';other.write_bytes(b'not the executing interpreter')
    monkeypatch.setattr(sys,'executable',str(other))
    with pytest.raises(EvidenceError,match='running interpreter'):gpu.host_runtime()


def mapping(path,inode=None):
    st=path.stat();return f'1000-2000 r-xp 00000000 {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x} {st.st_ino if inode is None else inode} {path}\n'


def test_mapped_runtime_includes_actual_os_math_cpp_loader_and_extra_extensions(tmp_path):
    from ovl_pipeline.canonical import file_hash
    names=['libc.so.6','libm.so.6','ld-linux-x86-64.so.2','libstdc++.so.6','libgcc_s.so.1','thirdparty_extension.so']
    text=''
    for name in names:
        path=tmp_path/name;path.write_bytes(b'\x7fELF'+name.encode());text+=mapping(path)*2
    got=gpu.mapped_libraries(maps=text)
    assert [v['name'] for v in got]==sorted(names)
    assert all(v['sha256']==file_hash(tmp_path/v['name']) for v in got)
    assert [v['name'] for v in gpu.mapped_libraries(os_only=True,maps=text)]==sorted(names[:-1])
    assert gpu.mapped_libraries(maps=''.join(reversed(text.splitlines(True))))==got
    old=got;(tmp_path/'libm.so.6').write_bytes(b'\x7fELFchanged math')
    assert gpu.mapped_libraries(maps=text)!=old


@pytest.mark.parametrize('damage',['deleted','replaced','wrong-device','non-elf','missing-math','malformed-inode'])
def test_mapped_os_libraries_cannot_be_missing_or_silently_replaced(tmp_path,damage):
    text=''
    for name in ('libc.so.6','libm.so.6','ld-linux-x86-64.so.2'):
        path=tmp_path/name;path.write_bytes(b'\x7fELF'+name.encode());text+=mapping(path)
    target=tmp_path/'libm.so.6'
    if damage=='deleted':target.unlink()
    elif damage=='replaced':
        other=tmp_path/'replacement';other.write_bytes(target.read_bytes());other.replace(target)
    elif damage=='wrong-device':text=text.replace(f'{os.major(target.stat().st_dev):02x}:{os.minor(target.stat().st_dev):02x}','ff:ff')
    elif damage=='non-elf':target.write_bytes(b'not an ELF')
    elif damage=='missing-math':text=''.join(line for line in text.splitlines(True) if 'libm.so' not in line)
    else:text=text.replace(str(target.stat().st_ino)+' '+str(target),'invalid '+str(target))
    with pytest.raises(EvidenceError):gpu.mapped_libraries(os_only=True,maps=text)
