"""Initialization orchestration under an explicit CPU substitute; no real CUDA credit."""
import copy
import pytest
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from ovl_pipeline import initialization
from ovl_pipeline.fixture import recipe
from ovl_pipeline.canonical import EvidenceError,digest,write_json


@pytest.fixture
def distinct_process_observations(monkeypatch):
    # Deliberately simulated alongside the CPU numerical runtime. Actual CUDA
    # acceptance requires two separate CLI processes; this is gate orchestration.
    counter=iter(range(100,110))
    monkeypatch.setattr(initialization,'process_identity',lambda:{'pid':next(counter),'boot_id':'synthetic','start_ticks':'1'})


def test_initial_state_is_regenerated_without_restoring_prover(cpu_runtime,distinct_process_observations,prepared,tmp_path,monkeypatch):
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size'])
    kernel={'schema':'ovl.gpu-kernel.v1','precision':'bf16'}
    value=initialization.record(directory/'wikipedia',r,kernel,tmp_path/'record')
    assert value['control']['global_step']==0 and 'pilot_cycle' not in value['control']
    import ovl_pipeline.state as state
    monkeypatch.setattr(state,'restore',lambda *a:pytest.fail('must not restore prover tensors'))
    result=initialization.verify(directory/'wikipedia',tmp_path/'record',digest(value),tmp_path/'verify')
    assert result['result']=='PASS' and result['prover_tensors_loaded_as_state'] is False
    assert result['production_admission']=='NOT_RUN' and result['independent_third_party'] is False

@pytest.mark.parametrize('change',['record-root','seed','checkpoint','extra-checkpoint-file','control','runtime'])
def test_initialization_rejects_substitution(cpu_runtime,distinct_process_observations,prepared,tmp_path,change):
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size'])
    value=initialization.record(directory/'wikipedia',r,{'schema':'ovl.gpu-kernel.v1','precision':'bf16'},tmp_path/'record')
    expected=digest(value)
    if change=='record-root':expected='0'*64
    elif change=='seed':value['recipe']['seed']+=1
    elif change=='checkpoint':
        p=tmp_path/'record/initial-state/state.safetensors';p.write_bytes(b'altered')
    elif change=='extra-checkpoint-file':(tmp_path/'record/initial-state/unregistered').write_bytes(b'not an input')
    elif change=='control':value['control']['cursor']=1
    else:value['environment']['compatible']={'different':'runtime'}
    if change not in ('record-root','checkpoint','extra-checkpoint-file'):
        write_json(tmp_path/'record/record.json',value);expected=digest(value)
    with pytest.raises((EvidenceError,FileNotFoundError)):initialization.verify(directory/'wikipedia',tmp_path/'record',expected,tmp_path/'verify')
    assert not (tmp_path/'verify/verification.json').exists()


def test_same_process_cannot_claim_fresh_initialization(cpu_runtime,prepared,tmp_path):
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size'])
    value=initialization.record(directory/'wikipedia',r,{'schema':'ovl.gpu-kernel.v1','precision':'bf16'},tmp_path/'record')
    with pytest.raises(EvidenceError,match='fresh process'):
        initialization.verify(directory/'wikipedia',tmp_path/'record',digest(value),tmp_path/'verify')
