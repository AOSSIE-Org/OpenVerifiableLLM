"""CPU record driver with explicit external endorsement substitutes; no paid work."""
import time
import pytest
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from test_production_chain import actual_artifacts
from ovl_pipeline import production_record as rec,production_trajectory as trajectory
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.training import code_root


def setup(prepared,tmp_path,monkeypatch):
    r,oldroot,reference,key,prover,streams=actual_artifacts(prepared,tmp_path)
    r['code_root']=code_root();r['runtime']['compatible_environment_sha256']=digest({'test_runtime':'CPU-substitute'})
    monkeypatch.setattr(rec,'verify_packet',lambda *a,**k:{'explicit-test-double':'not publisher evidence'})
    monkeypatch.setattr(rec,'object_at',lambda *a:r)
    def run(resume=False):return rec.record(None,None,None,None,None,streams,tmp_path/'record',key,tmp_path/'anchors',tmp_path/'policies.json',int(time.time())+600,resume=resume)
    return r,reference,key,streams,run


def test_complete_record_uses_same_fresh_trajectory_and_waits_at_every_boundary(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch);verified=[]
    def anchor(registration,root,envelopes,*a,**kwargs):
        b=envelopes[-1]['body'];reference=expected[b['index']]['body']
        assert b['control']==reference['control'] and b['kind']==reference['kind']
        assert b['checkpoint']['state_root']==reference['checkpoint']['state_root']
        verified.append(b['index']);return {'explicit-test-double':True,'boundaries_checked':len(envelopes)}
    monkeypatch.setattr(rec,'await_anchor',anchor)
    result=run();assert result['result']=='RECORDED_NOT_REPLAYED'
    assert verified==list(range(len(expected))) and result['public_prefixes_verified_this_process']==len(expected)
    assert result['recovery_checkpoints']>0
    log=read_json(tmp_path/'record/chain.json');assert log['complete'] is True
    with pytest.raises(EvidenceError,match='fresh record'):run()


@pytest.mark.parametrize('reject_index',[0,1])
def test_no_update_can_cross_unverified_public_boundary(cpu_runtime,prepared,tmp_path,monkeypatch,reject_index):
    r,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch);updates=[]
    original=trajectory.gpu.update
    def counted(model,opt,batch,control,*a,**k):
        if 'pilot_cycle' not in control:updates.append(control['global_step']+1)
        return original(model,opt,batch,control,*a,**k)
    monkeypatch.setattr(trajectory.gpu,'update',counted)
    def anchor(registration,root,envelopes,*a,**kwargs):
        if len(envelopes)-1==reject_index:raise EvidenceError('missing or forged public anchor')
        return {'explicit-test-double':True}
    monkeypatch.setattr(rec,'await_anchor',anchor)
    with pytest.raises(EvidenceError,match='public anchor'):run()
    assert len(updates)==expected[reject_index]['body']['control']['global_step']
    assert not(tmp_path/'record/record.json').exists()
    assert(tmp_path/f'record/boundary-{reject_index:05d}/checkpoint.json').exists()


def test_missing_anchor_deadline_preserves_waiting_checkpoint(tmp_path):
    with pytest.raises(EvidenceError,match='deadline'):
        rec.await_anchor({},'0'*64,[],tmp_path/'anchors',tmp_path/'external-policy.json',int(time.time())-1)


def test_operator_stop_only_returns_after_durable_boundary(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch)
    def anchor(*a,**kwargs):
        (tmp_path/'record/request-stop').touch();return {'explicit-test-double':True}
    monkeypatch.setattr(rec,'await_anchor',anchor)
    result=run();assert result['result']=='STOPPED' and result['control']['global_step']==0
    assert(tmp_path/'record/boundary-00000/checkpoint.json').exists()


@pytest.mark.parametrize('stop_kind',['initial','progress','base','transition','final'])
def test_restart_adopts_verified_public_prefix_and_matches_uninterrupted_states(cpu_runtime,prepared,tmp_path,monkeypatch,stop_kind):
    r,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch);stopped=False
    def anchor(registration,root,envelopes,*a,**kwargs):
        nonlocal stopped
        b=envelopes[-1]['body']
        if not stopped and b['kind']==stop_kind:
            stopped=True;raise EvidenceError('simulated interruption awaiting anchor')
        reference=expected[b['index']]['body']
        assert b['control']==reference['control'] and b['checkpoint']['state_root']==reference['checkpoint']['state_root']
        return {'explicit-test-double':True}
    monkeypatch.setattr(rec,'await_anchor',anchor)
    with pytest.raises(EvidenceError,match='simulated interruption'):run()
    assert stopped
    result=run(resume=True)
    assert result['result']=='RECORDED_NOT_REPLAYED' and result['recording_resumed'] is True
    final=read_json(tmp_path/'record/chain.json')['boundaries']
    assert len(final)==len(expected)
    assert [e['body']['checkpoint']['state_root'] for e in final]==[e['body']['checkpoint']['state_root'] for e in expected]


def test_resume_cannot_cross_missing_public_anchor(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch)
    def absent(*a,**k):raise EvidenceError('missing public anchor')
    monkeypatch.setattr(rec,'await_anchor',absent)
    with pytest.raises(EvidenceError):run()
    monkeypatch.setattr(rec,'resume_record',lambda *a:pytest.fail('must not restore before anchor'))
    with pytest.raises(EvidenceError,match='missing public anchor'):run(resume=True)
    assert not(tmp_path/'record/record.json').exists()


def test_stop_during_anchor_wait_returns_without_sleep(tmp_path):
    stop=tmp_path/'request-stop';stop.touch()
    with pytest.raises(EvidenceError,match='requested stop'):
        rec.await_anchor({},'0'*64,[],tmp_path/'anchors',tmp_path/'policies',int(time.time())+100,
                         stop_file=stop,sleep=lambda _:pytest.fail('do not wait while stop requested'))


def test_partial_uncommitted_checkpoint_is_preserved_and_regenerated(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch)
    monkeypatch.setattr(rec,'await_anchor',lambda *a,**k:{'explicit-test-double':True})
    original=rec.save_state;failed=False
    def interrupted(path,*args):
        nonlocal failed
        if path.name=='recovery-000000002' and not failed:
            failed=True;path.mkdir();(path/'state.safetensors').write_bytes(b'sole partial evidence')
            raise OSError('simulated interrupted checkpoint write')
        return original(path,*args)
    monkeypatch.setattr(rec,'save_state',interrupted)
    with pytest.raises(OSError,match='interrupted checkpoint'):run()
    assert failed
    result=run(resume=True)
    assert result['result']=='RECORDED_NOT_REPLAYED'
    preserved=list((tmp_path/'record-preserved-partials').glob('*/state.safetensors'))
    assert len(preserved)==1 and preserved[0].read_bytes()==b'sole partial evidence'
