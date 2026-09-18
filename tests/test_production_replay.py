"""Complete tiny CPU trajectories with explicit publisher/CUDA substitutes.

No actual production identity, provider admission, or CUDA replay credit.
"""
import pytest
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from test_production_chain import actual_artifacts,resign
from ovl_pipeline import production_replay as replay
from ovl_pipeline.canonical import EvidenceError,digest,inventory,write_json
from ovl_pipeline.training import code_root


def setup(prepared,tmp_path,monkeypatch):
    r,oldroot,envs,key,prover,streams=actual_artifacts(prepared,tmp_path)
    r['code_root']=code_root();r['runtime']['compatible_environment_sha256']=digest({'test_runtime':'CPU-substitute'})
    root=digest(r)
    for env in envs:env['body']['registration']=root
    envs[0]['body']['previous']=root;resign(envs,key)
    monkeypatch.setattr(replay,'authenticate',lambda *a:(r,envs,{'explicit-test-double':'publisher signatures not exercised'}))
    def run(output):return replay.replay(None,None,None,None,None,prover,None,None,streams,output)
    return r,envs,key,prover,run


def test_complete_both_phases_from_fresh_initialization_never_restore_prover(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,envs,key,prover,run=setup(prepared,tmp_path,monkeypatch)
    import ovl_pipeline.state as state
    import ovl_pipeline.training as training
    monkeypatch.setattr(state,'restore',lambda *a:pytest.fail('prover restore forbidden'))
    monkeypatch.setattr(training,'restore',lambda *a:pytest.fail('prover restore forbidden'))
    result=run(tmp_path/'verifier')
    assert result['result']=='PASS' and len(result['comparisons'])==len(envs)
    assert result['updates_recomputed']=={p:r['coverage'][p]['updates'] for p in r['coverage']}
    assert result['targets_recomputed']=={p:r['coverage'][p]['targets'] for p in r['coverage']}
    assert result['prover_checkpoints_restored'] is False and result['initial_state_regenerated'] is True
    assert result['independent_third_party'] is False and result['raw_transformation_reconstruction']=='NOT_RUN'
    assert len(list((tmp_path/'verifier').glob('verifier-boundary-*')))==len(envs)
    with pytest.raises(EvidenceError,match='fresh'):run(tmp_path/'verifier')


def test_forged_consistent_signed_state_fails_actual_continuous_replay(cpu_runtime,prepared,tmp_path,monkeypatch):
    from ovl_pipeline.state import read_state,unpack,pack,tensor_digest,state_root
    from safetensors.torch import save_file
    r,envs,key,prover,run=setup(prepared,tmp_path,monkeypatch)
    b=envs[1]['body'];path=prover/b['checkpoint_path'];md,ts=read_state(path,b['checkpoint'])
    obj=unpack(md['tree'],ts)
    name=next(iter(obj['model']));obj['model'][name]=obj['model'][name]+1
    tensors={};tree=pack(obj,tensors);md={'schema':'ovl.state.v1','tree':tree,'tensor_root':tensor_digest(tensors)}
    save_file(tensors,str(path/'state.safetensors'));write_json(path/'state.json',md)
    b['checkpoint']={'schema':'ovl.checkpoint.v1','state_root':state_root(md,tensors),
                     'files':inventory(path,['state.json','state.safetensors'])}
    write_json(path/'checkpoint.json',b['checkpoint']);resign(envs,key)
    with pytest.raises(EvidenceError,match='continuous replay state mismatch'):run(tmp_path/'verifier')
    assert not(tmp_path/'verifier/verification.json').exists()
    assert(tmp_path/'verifier/progress.json').exists()  # Preserve verified opening and failure prefix.


@pytest.mark.parametrize('change',['code','runtime','initialization'])
def test_source_runtime_and_initial_state_substitution_refused(cpu_runtime,prepared,tmp_path,monkeypatch,change):
    r,envs,key,prover,run=setup(prepared,tmp_path,monkeypatch)
    if change=='code':monkeypatch.setattr(replay,'code_root',lambda:'0'*64)
    elif change=='runtime':monkeypatch.setattr(replay.gpu,'environment',lambda c:{'compatible':{'altered':True}})
    else:
        original=replay.initialization.fresh
        def altered(*a):
            values=original(*a);values[2]['cursor']=1;return values
        monkeypatch.setattr(replay.initialization,'fresh',altered)
    with pytest.raises(EvidenceError):run(tmp_path/'verifier')
    assert not(tmp_path/'verifier/verification.json').exists()


def test_missing_identity_gate_never_initializes_device(tmp_path,monkeypatch):
    def refused(*a):raise EvidenceError('missing exact public endorsement')
    monkeypatch.setattr(replay,'authenticate',refused)
    monkeypatch.setattr(replay.initialization,'fresh',lambda *a:pytest.fail('no initialization before anchor checks'))
    with pytest.raises(EvidenceError,match='missing exact public endorsement'):
        replay.replay(None,None,None,None,None,None,None,None,None,tmp_path/'verifier')
    assert not(tmp_path/'verifier').exists()
