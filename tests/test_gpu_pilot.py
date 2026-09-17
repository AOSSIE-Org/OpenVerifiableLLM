"""Pilot orchestration on a deliberately substituted CPU test runtime.

These checks are not CUDA throughput, numerical exactness or production evidence.
"""
import copy

import pytest

from test_pipeline import prepared
from ovl_pipeline import gpu_pilot, training
from ovl_pipeline.canonical import EvidenceError, digest, write_json
from ovl_pipeline.fixture import recipe


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.setattr(gpu_pilot.gpu,"initialize",lambda r,c:training.initialize(r))
    monkeypatch.setattr(gpu_pilot.gpu,"flags",lambda:{"test_runtime":"CPU-substitute"})
    monkeypatch.setattr(gpu_pilot.gpu,"environment",lambda c:{"compatible":{"test_runtime":"CPU-substitute"},"physical_observation":{}})
    def update(model,opt,batch,control,total,config,*,expected_flags,metrics=None):
        assert expected_flags=={"test_runtime":"CPU-substitute"}
        return training.update(model,opt,batch,control,total,metrics=metrics)
    monkeypatch.setattr(gpu_pilot.gpu,"update",update)
    monkeypatch.setattr(gpu_pilot.torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(gpu_pilot.torch.cuda,"empty_cache",lambda:None)


@pytest.mark.parametrize("phase",["wikipedia","conversation"])
def test_continuous_pilot_replay_and_separately_labeled_resume(cpu_runtime,prepared,tmp_path,monkeypatch,phase):
    directory,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"])
    config={"schema":"ovl.gpu-kernel.v1","precision":"bf16"}
    record=gpu_pilot.record(directory/phase,r,config,tmp_path/"record",updates=9,warmup_updates=2,checkpoint_every=2)
    assert record["result"]=="RECORDED_NOT_REPLAYED" and record["eligible_duration_for_forecast"] is False
    assert record["production_admission"]=="NOT_RUN"
    assert 0 < record["measured_full_batch_updates"] <= record["updates"]
    with monkeypatch.context() as m:
        m.setattr(gpu_pilot,"restore",lambda *args:pytest.fail("continuous replay must not load a prover checkpoint"))
        replay=gpu_pilot.replay(directory/phase,tmp_path/"record",digest(record),tmp_path/"replay")
    assert replay["updates_recomputed"]==9 and len(replay["compared"])==len(record["boundaries"])
    assert replay["scope"]=="fresh-initialization-continuous-pilot-replay"
    assert replay["timed_checkpoints"]==record["timed_checkpoints"]==5
    assert replay["measured_targets"]==record["measured_targets"]
    assert replay["measured_full_batch_updates"]==record["measured_full_batch_updates"]
    assert replay["measured_ms"]>0 and replay["setup_including_warmup_ms"]>0
    assert replay["eligible_for_forecast_comparison"] is False
    resume=gpu_pilot.replay(directory/phase,tmp_path/"record",digest(record),tmp_path/"resume",resume_from=2)
    assert resume["updates_recomputed"]==5 and resume["scope"]=="training-resume-continuation-probe"
    assert resume["compared"][-1]["state_root"]==replay["compared"][-1]["state_root"]
    assert resume["eligible_for_forecast_comparison"] is False
    assert replay["independent_third_party"] is False and resume["production_training_coverage"]=="NOT_RUN"


def test_record_hash_and_broken_ancestry_refused(cpu_runtime,prepared,tmp_path):
    directory,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"])
    record=gpu_pilot.record(directory/"wikipedia",r,{"schema":"ovl.gpu-kernel.v1","precision":"fp32"},tmp_path/"record",updates=5,checkpoint_every=2)
    with pytest.raises(EvidenceError,match="selected digest"):
        gpu_pilot.replay(directory/"wikipedia",tmp_path/"record","0"*64,tmp_path/"bad")
    record["boundaries"][1]["previous"]="0"*64;write_json(tmp_path/"record/record.json",record)
    with pytest.raises(EvidenceError,match="ancestry"):
        gpu_pilot.replay(directory/"wikipedia",tmp_path/"record",digest(record),tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_wrong_checkpoint_fails_continuous_comparison(cpu_runtime,prepared,tmp_path):
    directory,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"])
    value=gpu_pilot.record(directory/"conversation",r,{"schema":"ovl.gpu-kernel.v1","precision":"fp32"},tmp_path/"record",updates=5,checkpoint_every=2)
    path=tmp_path/"record/boundary-00001/state.safetensors";data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(EvidenceError,match="hash mismatch"):
        gpu_pilot.replay(directory/"conversation",tmp_path/"record",digest(value),tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_changed_kernel_code_environment_and_schedule_are_not_replayable(cpu_runtime,prepared,tmp_path,monkeypatch):
    directory,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"])
    value=gpu_pilot.record(directory/"wikipedia",r,{"schema":"ovl.gpu-kernel.v1","precision":"fp32"},tmp_path/"record",updates=5,checkpoint_every=2)
    changed=copy.deepcopy(value);changed["settings"]["code_root"]="0"*64
    write_json(tmp_path/"record/record.json",changed)
    with pytest.raises(EvidenceError,match="code differs"):
        gpu_pilot.replay(directory/"wikipedia",tmp_path/"record",digest(changed),tmp_path/"bad")
    changed=copy.deepcopy(value);changed["boundaries"].pop(1);write_json(tmp_path/"record/record.json",changed)
    with pytest.raises(EvidenceError):gpu_pilot.replay(directory/"wikipedia",tmp_path/"record",digest(changed),tmp_path/"bad")
    write_json(tmp_path/"record/record.json",value)
    monkeypatch.setattr(gpu_pilot.gpu,"environment",lambda c:{"compatible":{"test_runtime":"changed"},"physical_observation":{}})
    with pytest.raises(EvidenceError,match="environment differs"):
        gpu_pilot.replay(directory/"wikipedia",tmp_path/"record",digest(value),tmp_path/"bad")


def test_short_timing_and_unbounded_checkpoint_schedule_refused(prepared,tmp_path):
    directory,manifest=prepared;r=recipe(manifest["tokenizer"]["vocab_size"]);c={"schema":"ovl.gpu-kernel.v1","precision":"bf16"}
    with pytest.raises(EvidenceError):gpu_pilot.record(directory/"wikipedia",r,c,tmp_path/"bad",seconds=599)
    with pytest.raises(EvidenceError):gpu_pilot.record(directory/"wikipedia",r,c,tmp_path/"bad",updates=10000,checkpoint_every=1)
    assert not (tmp_path/"bad").exists()
