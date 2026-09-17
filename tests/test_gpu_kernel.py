"""CPU checks of shared-kernel behavior; no claim of measured CUDA exactness."""
import math
import struct

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
