"""Actual safe snapshots/downloaded anchors; service/provider/signature doubles.

No CUDA or production acceptance. Existing publisher fixture executes real file
uploads/download checks against an explicit fake provider before these tests.
"""
from pathlib import Path
import shutil
import sys
import time
import pytest
import production_boundary_poll as m
from test_pipeline import prepared
from test_progress_dispatch import setup as publication
from test_pod_transfer import setup as ssh
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json,verify_inventory


def fixture(prepared,tmp_path,monkeypatch,*,retained_store=None):
    source=tmp_path/'reference';source.mkdir()
    run,provider,_=publication(prepared,source,monkeypatch,persistent=True)
    ack=run(0)
    spec=read_json(source/'service-state-00000/selection.json')
    arguments={k:spec['arguments'][k] for k in ('packet','registration-bundle','production-policy','source-policy','source-checkout','config')}
    r=read_json(Path(arguments['packet'])/'registration.json')
    target=tmp_path/'transport';target.mkdir();transport,remote,_,_=ssh(target)
    shutil.copytree(source/'numerical/checkpoints',remote,dirs_exist_ok=True)
    job='a'*64;root=digest(r);start_epoch=int(time.time());exports=[];starts=[]
    class Health:
        plan={'input':{'now_epoch':start_epoch-10},'request_checkpoint_epoch':start_epoch+1600,'external_terminate_epoch':start_epoch+1800}
        def contract(self,j):
            assert j==job
            return {'kind':'production-record','registration_sha256':root,
                    'outputs':[{'root':transport.profile['remote_root'],'profile_sha256':digest(transport.profile)}]}
        def active(self,j):assert j==job
        def now(self):return int(time.time())
        def bytes(self,*a,**k):pass
        def write(self,*a):pass
        def publication(self,*a):pass
        def exported_files(self,j,directory,files):
            assert j==job;verify_inventory(directory,files);exports.append(digest(files))
    h=Health();policy={'schema':'ovl.production-boundary-publication-policy.v1','boundary_seconds':1200,
                       'snapshot_seconds':120,'delivery_seconds':60,'python':str(Path(sys.executable).resolve())}
    output=tmp_path/'coordinator'
    def start(selected,expected,directory):
        m.publisher.validate(selected,expected)
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if not(directory/'result.json').exists():
            starts.append(expected)
            dest=Path(selected['arguments']['output'])
            shutil.copytree(source/'publication/boundary-00000',dest)
            # The reference fixture's request-commit double does not write this
            # receipt; add the same explicit public revision it returned.
            write_json(dest/'git-request/public-commit.json',{'revision':ack['policy']['source_revision']})
            write_json(directory/'result.json',{'schema':'ovl.publisher-service-result.v1','selection_sha256':expected,
                'ack_sha256':digest(ack),'ack_path':str(dest/'ack.json'),'scope':'explicit completed-service double'})
        return {'observation':{'ActiveState':'inactive'}}
    monkeypatch.setattr(m.publisher,'start_or_adopt',start)
    def hook():return m.BoundaryPublisher(r,job,h,tmp_path/'health.json',transport,output,arguments,policy,retained_store=retained_store)
    def advance():
        from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
        run(1,Path(ack['anchor_directory']),[ProgressPublisherPolicy(**ack['policy'])])
        for name in ('chain.json','awaiting-anchor.json'):
            shutil.copyfile(source/'numerical/checkpoints'/name,remote/name)
    hook.advance=advance
    return hook,remote,output,h,starts,provider


def test_complete_state_publication_and_external_policy_handoff_adopt_once(prepared,tmp_path,monkeypatch):
    hook,remote,out,h,starts,provider=fixture(prepared,tmp_path,monkeypatch)
    a=hook().poll();assert a['index']==0
    assert len(read_json(remote/'external-progress-policies.json'))==1
    old=read_json(out/'boundaries/boundary-00000/intent.json')
    assert hook().poll()==a and len(starts)==1 and provider.commits==2
    assert read_json(out/'boundaries/boundary-00000/intent.json')==old


@pytest.mark.parametrize('damage',['registration','signature','state','deadline','policy','service-result','source'])
def test_broken_boundary_fails_without_policy_delivery(prepared,tmp_path,monkeypatch,damage):
    hook,remote,out,h,starts,_=fixture(prepared,tmp_path,monkeypatch)
    original=m.publisher.start_or_adopt
    if damage=='registration':
        v=read_json(remote/'awaiting-anchor.json');v['registration_sha256']='f'*64;write_json(remote/'awaiting-anchor.json',v)
    elif damage=='signature':
        v=read_json(remote/'chain.json');v['boundaries'][0]['signature']='0'*128;write_json(remote/'chain.json',v)
    elif damage=='state':(remote/'boundary-00000/state.safetensors').write_bytes(b'altered')
    else:
        def changed(spec,expected,directory):
            value=original(spec,expected,directory);p=Path(spec['arguments']['output'])
            if damage=='policy':
                v=read_json(p/'operator-policy.json');v['source_revision']='f'*40;write_json(p/'operator-policy.json',v)
            elif damage=='service-result':
                v=read_json(Path(directory)/'result.json');v['ack_sha256']='f'*64;write_json(Path(directory)/'result.json',v)
            elif damage=='deadline':
                h.now=lambda:spec['deadline_epoch']+61
            else:
                (Path(spec['arguments']['source-checkout'])/'unexpected').write_text('changed')
            return value
        monkeypatch.setattr(m.publisher,'start_or_adopt',changed)
    with pytest.raises(EvidenceError):hook().poll()
    assert not(remote/'external-progress-policies.json').exists()


@pytest.mark.parametrize('advanced',[False,True])
def test_lost_policy_write_response_reconciles_without_new_publication(prepared,tmp_path,monkeypatch,advanced):
    hook,remote,out,h,starts,_=fixture(prepared,tmp_path,monkeypatch)
    first=hook();original=first.transport.put;failed=[False]
    def put(name,*args,**kwargs):
        result=original(name,*args,**kwargs)
        if name=='external-progress-policies.json' and not failed[0]:
            failed[0]=True;raise EvidenceError('explicit lost response after atomic write')
        return result
    first.transport.put=put
    with pytest.raises(EvidenceError,match='lost response'):first.poll()
    assert (remote/'external-progress-policies.json').exists()
    if advanced:hook.advance()
    intent=read_json(out/'boundaries/boundary-00000/intent.json')
    result=hook().poll();assert result['index']==0 and len(starts)==1
    assert read_json(out/'boundaries/boundary-00000/intent.json')==intent
    assert len(list((out/'boundaries/boundary-00000').glob('reconcile-*')))==1


def test_snapshot_retry_preserves_partial_and_original_deadline(prepared,tmp_path,monkeypatch):
    hook,remote,out,h,starts,_=fixture(prepared,tmp_path,monkeypatch)
    original=m.snapshot;limits=[]
    def failed(*a,**kw):
        if limits:
            limits.append(a[4]);return original(*a,**kw)
        from pod_transfer import TransientTransportError
        limits.append(a[4]);Path(a[3]).mkdir();(Path(a[3])/'failure-note').write_bytes(b'preserved diagnostic')
        error=TransientTransportError('explicit interrupted snapshot')
        error.transfer_counts={'bytes_sent':0,'bytes_received':0};error.immutable_download_bytes=0
        raise error
    monkeypatch.setattr(m,'snapshot',failed)
    assert hook().poll()['index']==0 and len(starts)==1 and limits[0]==limits[1]
    old=read_json(out/'boundaries/boundary-00000/intent.json')
    assert hook().poll()['index']==0 and len(starts)==1 and len(limits)==2
    assert (out/'boundaries/boundary-00000/snapshot/failure-note').read_bytes()==b'preserved diagnostic'
    assert read_json(out/'boundaries/boundary-00000/intent.json')==old


def test_two_partial_snapshots_do_not_get_a_third_copy_attempt(prepared,tmp_path,monkeypatch):
    hook,remote,out,h,starts,_=fixture(prepared,tmp_path,monkeypatch);calls=[]
    def failed(*a,**kw):
        from pod_transfer import TransientTransportError
        calls.append(a[4]);Path(a[3]).mkdir();(Path(a[3])/'failure-note').write_bytes(b'preserved diagnostic')
        error=TransientTransportError('explicit interrupted snapshot')
        error.transfer_counts={'bytes_sent':0,'bytes_received':0};error.immutable_download_bytes=0
        raise error
    monkeypatch.setattr(m,'snapshot',failed)
    for _ in range(3):
        with pytest.raises(EvidenceError):hook().poll()
    assert len(calls)==2 and calls[0]==calls[1] and not starts


@pytest.mark.parametrize('state',['complete','bad-result','running'])
def test_completed_service_adopts_within_delivery_window_without_late_activity_credit(prepared,tmp_path,monkeypatch,state):
    from production_health import ProductionHealth
    hook,remote,out,h,starts,provider=fixture(prepared,tmp_path,monkeypatch)
    first=hook();clock=[h.now()];h.now=lambda:clock[0]
    h.plan['provider_terminate_epoch']=h.plan['external_terminate_epoch']
    observed=[]
    def activity(*args):
        observed.append(args)
        return ProductionHealth.publication(h,*args)
    h.publication=activity
    original=m.publisher.start_or_adopt;selected=[]
    def service(spec,expected,directory):
        value=original(spec,expected,directory);selected.append(spec)
        clock[0]=spec['deadline_epoch']+1  # Original delivery window remains open.
        destination=Path(spec['arguments']['output'])
        assert list((destination/'activity').glob('*.json'))
        result=Path(directory)/'result.json'
        if state=='running':result.unlink();return {'observation':{'ActiveState':'active'}}
        if state=='bad-result':
            changed=read_json(result);changed['ack_sha256']='f'*64;write_json(result,changed)
        return value
    monkeypatch.setattr(m.publisher,'start_or_adopt',service)
    if state=='complete':
        assert first.poll()['index']==0 and observed==[]
        assert (remote/'external-progress-policies.json').exists()
        # The real health method still rejects late telemetry. Completion does
        # not create a liveness exception or revive an expired worker deadline.
        with pytest.raises(EvidenceError,match='publication deadline reached'):
            ProductionHealth.publication(h,first.job,tmp_path,selected[0]['deadline_epoch'],{})
    else:
        with pytest.raises(EvidenceError,match='publication deadline reached' if state=='running' else 'publisher result differs'):
            first.poll()
        assert bool(observed)==(state=='running')
        assert not(remote/'external-progress-policies.json').exists()
    intent=read_json(out/'boundaries/boundary-00000/intent.json')
    assert selected[0]['deadline_epoch']==intent['deadline_epoch']-first.policy['delivery_seconds']
