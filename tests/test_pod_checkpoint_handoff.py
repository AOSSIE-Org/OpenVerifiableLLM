"""Actual tiny signed CPU states and SSH subprocess transfers; publisher doubles explicit."""
from dataclasses import asdict,replace
from pathlib import Path
import shutil
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_checkpoint_handoff as m
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from test_production_chain import actual_artifacts
from test_pod_transfer import setup as ssh
from test_progress_dispatch import setup as publisher
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
from ovl_pipeline.production_record import await_anchor


def paused(prepared,tmp_path):
    t,remote,calls,processes=ssh(tmp_path)
    numerical=tmp_path/'numerical';numerical.mkdir()
    r,root,envelopes,key,chain,streams=actual_artifacts(prepared,numerical)
    e=envelopes[0];body=e['body']
    write_json(chain/'chain.json',{'schema':'ovl.production-chain.v1','complete':False,'boundaries':[e]})
    write_json(chain/'awaiting-anchor.json',{'schema':'ovl.awaiting-public-progress.v1','registration_sha256':root,'index':0,
               'boundary_sha256':digest(e),'checkpoint_path':body['checkpoint_path'],'checkpoint':body['checkpoint']})
    shutil.copytree(chain,remote,dirs_exist_ok=True)
    return t,remote,r,root,chain,calls


def test_complete_signed_state_export_and_recheck_before_receipt(prepared,tmp_path):
    t,remote,r,root,chain,calls=paused(prepared,tmp_path)
    out=tmp_path/'export';receipt=m.snapshot(t,r,root,out,int(time.time())+60)
    assert receipt['result']=='PASS' and receipt['training_replay']=='NOT_RUN' and receipt['workload_complete'] is False
    assert receipt['checkpoint']==read_json(chain/'boundary-00000/checkpoint.json')
    assert all((out/'boundary-00000'/f['path']).read_bytes()==(chain/'boundary-00000'/f['path']).read_bytes() for f in receipt['files'])
    assert len(receipt['transfers'])==3
    with pytest.raises(FileExistsError):m.snapshot(t,r,root,out,int(time.time())+60)


@pytest.mark.parametrize('damage',['signature','registration','waiting','state','state-marker','metadata-race','missing-state'])
def test_wrong_or_changing_snapshot_never_reports_verified_export(prepared,tmp_path,damage):
    t,remote,r,root,chain,calls=paused(prepared,tmp_path)
    if damage=='signature':
        c=read_json(remote/'chain.json');c['boundaries'][0]['signature']='0'*128;write_json(remote/'chain.json',c)
    elif damage=='registration':root='0'*64
    elif damage=='waiting':
        w=read_json(remote/'awaiting-anchor.json');w['index']=1;write_json(remote/'awaiting-anchor.json',w)
    elif damage in ('state','state-marker'):
        (remote/'boundary-00000'/('state.safetensors' if damage=='state' else 'checkpoint.json')).write_bytes(b'altered')
    elif damage=='missing-state':(remote/'boundary-00000/state.safetensors').unlink()
    else:
        original=t.get
        def get(name,*args,**kwargs):
            result=original(name,*args,**kwargs)
            if name.endswith('state.safetensors'):
                w=read_json(remote/'awaiting-anchor.json');w['index']=1;write_json(remote/'awaiting-anchor.json',w)
            return result
        t.get=get
    out=tmp_path/'export'
    with pytest.raises(EvidenceError):m.snapshot(t,r,root,out,int(time.time())+60)
    assert not(out/'export.json').exists() and out.is_dir()


def published(prepared,tmp_path,monkeypatch):
    # Existing publisher fixture performs actual closed checkpoint uploads and
    # fresh downloads using an explicit fake HF service/Actions/Sigstore adapter.
    run,provider,committed=publisher(prepared,tmp_path,monkeypatch)
    ack=run(0)
    transfer=tmp_path/'transport';transfer.mkdir();t,remote,calls,processes=ssh(transfer)
    chain=tmp_path/'numerical/checkpoints';shutil.copytree(chain,remote,dirs_exist_ok=True)
    r=read_json(tmp_path/'source/packet/registration.json') if (tmp_path/'source/packet').exists() else None
    # Fixture's actual packet path is obtained from the retained file, scoped to
    # the test's owned source directory only.
    if r is None:r=read_json(next((tmp_path/'source').rglob('registration.json')))
    root=digest(r);out=tmp_path/'export';m.snapshot(t,r,root,out,int(time.time())+60)
    policy=ProgressPublisherPolicy(**read_json(tmp_path/'publication/boundary-00000/operator-policy.json'))
    return t,remote,r,root,out,ack,[policy],calls


def test_verified_public_ack_then_recorder_reverification_and_idempotent_retry(prepared,tmp_path,monkeypatch):
    t,remote,r,root,out,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    sent=[];put=t.put
    def counted(name,*a,**k):sent.append(name);return put(name,*a,**k)
    t.put=counted
    result=m.deliver(t,r,root,out,ack,policies,tmp_path/'delivery',int(time.time())+60)
    assert result['result']=='PASS' and result['training_replay']=='NOT_RUN'
    assert read_json(remote/'external-progress-policies.json')==[asdict(p) for p in policies]
    envs=read_json(out/'chain.json')['boundaries']
    assert await_anchor(r,root,envs,remote/'anchors',remote/'external-progress-policies.json',int(time.time())+30)['result']=='PASS'
    assert len([n for n in sent if n.startswith('anchors/')])==2
    sent.clear()
    assert m.deliver(t,r,root,out,ack,policies,tmp_path/'redelivery',int(time.time())+60)['result']=='PASS'
    assert not [n for n in sent if n.startswith('anchors/')]


@pytest.mark.parametrize('damage',['saved-pass','policy','anchor','local-state','waiting','rollback','partial-put'])
def test_failed_handoff_never_installs_new_policy(prepared,tmp_path,monkeypatch,damage):
    t,remote,r,root,out,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    write_json(remote/'external-progress-policies.json',[])
    if damage=='saved-pass':ack={'result':'PASS','schema':ack['schema']}
    elif damage=='policy':policies=[replace(policies[0],source_revision='f'*40)]
    elif damage=='anchor':write_json(Path(ack['anchor_directory'])/'progress-00000/statement.json',{'result':'PASS'})
    elif damage=='local-state':(out/'boundary-00000/state.safetensors').write_bytes(b'changed')
    elif damage=='waiting':
        w=read_json(remote/'awaiting-anchor.json');w['index']=5;write_json(remote/'awaiting-anchor.json',w)
    elif damage=='rollback':write_json(remote/'external-progress-policies.json',[asdict(policies[0])]*2)
    else:
        original=t.put
        def put(name,*args,**kwargs):
            if name.endswith('statement.sigstore.json'):raise EvidenceError('explicit interrupted network write')
            return original(name,*args,**kwargs)
        t.put=put
    previous=(remote/'external-progress-policies.json').read_bytes()
    with pytest.raises(EvidenceError):m.deliver(t,r,root,out,ack,policies,tmp_path/'delivery',int(time.time())+60)
    assert (remote/'external-progress-policies.json').read_bytes()==previous
    assert not(tmp_path/'delivery/delivery.json').exists()
    if damage=='partial-put':assert (remote/'anchors/progress-00000/statement.json').exists()


@pytest.mark.parametrize('damage',['altered','foreign'])
def test_remote_anchor_inventory_mismatch_is_preserved_and_refused(prepared,tmp_path,monkeypatch,damage):
    t,remote,r,root,out,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    m.deliver(t,r,root,out,ack,policies,tmp_path/'first-delivery',int(time.time())+60)
    target=remote/('anchors/progress-00000/statement.json' if damage=='altered' else 'anchors/unselected.json')
    target.write_bytes(b'preserve mismatched peer bytes')
    previous=(remote/'external-progress-policies.json').read_bytes()
    with pytest.raises(EvidenceError,match='remote anchor bytes differ'):
        m.deliver(t,r,root,out,ack,policies,tmp_path/'second-delivery',int(time.time())+60)
    assert target.read_bytes()==b'preserve mismatched peer bytes'
    assert (remote/'external-progress-policies.json').read_bytes()==previous
    assert not(tmp_path/'second-delivery/delivery.json').exists()


def test_partial_anchor_retry_only_transfers_missing_bytes(prepared,tmp_path,monkeypatch):
    t,remote,r,root,out,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    original=t.put
    def interrupted(name,*args,**kwargs):
        if name.endswith('statement.sigstore.json'):raise EvidenceError('explicit transfer interruption')
        return original(name,*args,**kwargs)
    t.put=interrupted
    with pytest.raises(EvidenceError,match='explicit transfer interruption'):
        m.deliver(t,r,root,out,ack,policies,tmp_path/'interrupted',int(time.time())+60)
    sent=[]
    def counted(name,*args,**kwargs):sent.append(name);return original(name,*args,**kwargs)
    t.put=counted
    m.deliver(t,r,root,out,ack,policies,tmp_path/'retry',int(time.time())+60)
    assert sent==['anchors/progress-00000/statement.sigstore.json','external-progress-policies.json']
    envs=read_json(out/'chain.json')['boundaries']
    assert await_anchor(r,root,envs,remote/'anchors',remote/'external-progress-policies.json',int(time.time())+30)['result']=='PASS'


def test_separate_control_profile_preserves_fresh_record_and_exports_real_initial_state(cpu_runtime,prepared,tmp_path,monkeypatch):
    import shlex
    import subprocess
    from pod_transfer import Transport,REMOTE_GET,REMOTE_PUT
    from test_production_record import setup as record_setup
    from ovl_pipeline import production_record as rec
    transport_dir=tmp_path/'transport';transport_dir.mkdir()
    control,control_dir,calls,processes=ssh(transport_dir)
    numerical=tmp_path/'numerical';numerical.mkdir()
    r,reference,key,streams,run=record_setup(prepared,numerical,monkeypatch)
    record_dir=numerical/'record'
    profile=dict(control.profile,remote_root=control.profile['remote_root']+'-record')
    def local_ssh(command,**kwargs):
        args=shlex.split(command[-1]);assert args.pop(0)=='exec'
        assert args[:2]==['/usr/bin/python3','-c'] and args[2] in (REMOTE_GET,REMOTE_PUT)
        assert args[3]==profile['remote_root'];args[0]=sys.executable;args[3]=str(record_dir)
        return subprocess.Popen(args,**kwargs)
    recording=Transport(profile,control.key,control.known,popen=local_ssh)
    descriptor=tmp_path/'control-input.json';write_json(descriptor,{'explicit':'test-control-only'})
    control.put('jobs/test/job.json',descriptor,int(time.time())+30)
    assert (control_dir/'jobs/test/job.json').is_file() and not record_dir.exists()
    exports=[]
    def anchor(registration,root,envelopes,*args,**kwargs):
        receipt=m.snapshot(recording,registration,root,tmp_path/'initial-export',int(time.time())+60)
        exports.append(receipt)
        recording.put('request-stop',descriptor,int(time.time())+30,replace=True)
        return {'explicit-test-double':'publisher verification excluded from this CPU transport fixture'}
    monkeypatch.setattr(rec,'await_anchor',anchor)
    result=run()
    assert result['result']=='STOPPED' and result['control']['global_step']==0
    assert len(exports)==1 and exports[0]['checkpoint']['state_root']==reference[0]['body']['checkpoint']['state_root']
    assert (record_dir/'request-stop').is_file() and not(control_dir/'request-stop').exists()


def test_changed_registration_between_signature_check_and_load_is_rejected(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(m,'verify_packet',lambda *a,**k:{'registration_sha256':'a'*64})
    monkeypatch.setattr(m,'object_at',lambda *a:{'changed':'registration'})
    with pytest.raises(EvidenceError,match='registration changed'):
        m.authenticated(tmp_path,tmp_path,SimpleNamespace(statement_sha256='a'*64),None,tmp_path)
