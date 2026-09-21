"""Actual terminal worker and retained bytes; failed qualification stays failed."""
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

import failed_stage_retention as m
import pod_versioned_export as versioned
import run_workload_stage as stage
from sustained_pilot_abort import stop_and_retain
from pod_transfer import EvidenceError,TransientTransportError,RangeRecoveryExhausted
from ovl_pipeline.canonical import digest,file_hash,read_json,write_json
from ovl_pipeline.supervision import Journal
from workload_health import Health
from test_workload_stage import staged,intent
from test_pod_job_worker import exited


def failed(tmp_path,health,t,remote,job,root,worker,worker_root):
    out=tmp_path/'stage';store=tmp_path/'store'
    health.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':health.pod,'kind':'pilot'})
    stage.launch(t,job,root,worker,worker_root,out/'launch',int(time.time())+30)
    assert exited(remote/'jobs'/root)['exit_code']==0
    original=t.get
    def fail(name,destination,expected,deadline,**kwargs):
        data=(remote/name).read_bytes();partial=destination.with_name(destination.name+'.partial')
        partial.write_bytes(data[:5]);ranges=partial.with_name(partial.name+'.ranges');ranges.mkdir()
        attempt=ranges/'attempt-00000.partial';attempt.write_bytes(data[5:9])
        write_json(ranges/'attempt-00000.json',{'offset':5,'bytes_requested':8,'bytes_received':4,
            'saved_bytes':4,'deadline_epoch':deadline,'result':'FAILED','error_type':'TransientTransportError',
            'partial_sha256':file_hash(attempt)})
        cause=TransientTransportError('synthetic exhausted delivery');cause.transfer_counts={'bytes_sent':0,'bytes_received':4}
        raise RangeRecoveryExhausted('synthetic finite retry exhaustion') from cause
    t.get=fail
    try:
        versioned.checkpoint_export(t,'output',store,out/'export-001',int(time.time())+30,expected_files=None)
    except RangeRecoveryExhausted as error:
        failure={'schema':'ovl.sustained-stage-dispatch-failure.v1','job_sha256':root,
                 'error_type':type(error).__name__,'scope':'synthetic failed export'}
        write_json(out/'dispatch-failure.json',failure);m.classify(error,failure,t,out,health.now())
    finally:t.get=original
    return out,store


def arguments(tmp_path,t,h,job,root,worker,worker_root,out,store):
    return (t,h,job,root,worker,worker_root,out,tmp_path/'health.json',tmp_path/'stop.json','d'*64),{
        'initial_retention':SimpleNamespace(store=store),
        'failure_limits':{'maximum_bytes':1024**2,'maximum_uncached_bytes':1024**2,'export_seconds':1200},'sleep':lambda _:time.sleep(.02)}


def test_terminal_backup_preserves_failure_partials_and_never_restarts_or_promotes(tmp_path):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        original=(out/'dispatch-failure.json').read_bytes();old=list(store.glob('incoming/*/verified.partial'))[0].read_bytes()
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        with pytest.raises(EvidenceError,match='retention only'):stage.run_stage(*args)
        result=stop_and_retain(*args,**kw)
        assert result['exit']['exit_code']==0 and h.jobs[root]['finished'] and not h.complete
        assert 'qualification FAILED' in result['scope'] and not(out/'stage-result.json').exists()
        assert read_json(out/'failure-retention/retention.json')['qualification']=='FAILED'
        assert (out/'dispatch-failure.json').read_bytes()==original
        assert list(store.glob('incoming/*/verified.partial'))[0].read_bytes()==old
        assert (Path(result['exports'][1]['directory'])/'payload').read_bytes()==(remote/'output/payload').read_bytes()
        before=len(calls);assert stop_and_retain(*args,**kw)==result and len(calls)==before
        with pytest.raises(EvidenceError,match='retention only'):stage.run_stage(*args)
        assert len([c for c in calls if ' start ' in c[-1]])==1


@pytest.mark.parametrize('damage',['aggregate','failed-range','record','missing-selection','identity','deadline','budget'])
def test_failure_backup_rejects_contradictions_and_bounds_without_health_credit(tmp_path,damage):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        attempt=next(store.glob('incoming/*'));args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        if damage=='aggregate':(attempt/'verified.partial').write_bytes(b'WRONG')
        elif damage=='failed-range':
            p=attempt/'verified.partial.ranges/attempt-00000.partial';p.write_bytes(b'BAD!')
            receipt=p.with_suffix('.json');v=read_json(receipt);v['partial_sha256']=file_hash(p);write_json(receipt,v)
        elif damage=='record':
            p=attempt/'verified.partial.ranges/attempt-00000.json';v=read_json(p);v['partial_sha256']='f'*64;write_json(p,v)
        elif damage=='missing-selection':(attempt/'download-selection.json').unlink()
        elif damage=='identity':t.profile['pod_id']='wrong-pod'
        elif damage=='deadline':
            now=h.now();h.now=lambda:now+1801
        else:kw['failure_limits']['maximum_bytes']=1
        with pytest.raises((EvidenceError,FileNotFoundError)):stop_and_retain(*args,**kw)
        assert not h.jobs[root]['finished'] and not h.complete
        assert not(out/'failure-retention/retention.json').exists()
        assert (out/'dispatch-failure.json').exists()
        assert len([c for c in calls if ' start ' in c[-1]])==1


@pytest.mark.parametrize('damage',['fatal','authentication','cleanup','missing-counts','unclassified'])
def test_only_closed_transient_exhaustion_can_select_a_separate_backup(tmp_path,damage):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path);out=tmp_path/'stage';out.mkdir()
    cause=TransientTransportError('synthetic transport failure');cause.transfer_counts={'bytes_sent':0,'bytes_received':1}
    error=RangeRecoveryExhausted('synthetic exhausted transfer');error.__cause__=cause
    if damage in ('fatal','authentication'):error.__cause__=EvidenceError('synthetic strict failure')
    elif damage=='cleanup':cause.transport_cleanup_diagnostic={'exception_class':'PermissionError'}
    elif damage=='missing-counts':del cause.transfer_counts
    else:error=TransientTransportError('synthetic unexhausted failure')
    m.classify(error,{'job_sha256':root},t,out,int(time.time()))
    assert not(out/'failure-retention-eligibility.json').exists() and not calls


def test_backup_restart_preserves_deadline_and_never_resets_incomplete_attempt(tmp_path,monkeypatch):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store);original=m.export
        def interrupted(transport,name,store,dest,*a,**k):
            dest.mkdir();(dest/'preserved.partial').write_bytes(b'private synthetic partial')
            raise EvidenceError('synthetic interruption')
        monkeypatch.setattr(m,'export',interrupted)
        with pytest.raises(EvidenceError,match='synthetic interruption'):stop_and_retain(*args,**kw)
        saved=(out/'failure-retention/intent.json').read_bytes();monkeypatch.setattr(m,'export',original)
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):stop_and_retain(*args,**kw)
        assert (out/'failure-retention/intent.json').read_bytes()==saved and not h.jobs[root]['finished']
        kw['failure_limits']['export_seconds']+=1
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):stop_and_retain(*args,**kw)


def test_controller_stop_shortens_backup_and_cannot_disappear(tmp_path):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        selected,check=m.prepare(t,h,job,root,worker,worker_root,out,tmp_path/'stop.json','d'*64,kw['failure_limits'],store)
        original=selected['binding']['deadline_epoch'];now=h.now()
        write_json(tmp_path/'stop.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
            'pod_id':h.pod,'observed_epoch':now,'reasons':['synthetic stop']})
        assert check()==min(original,now+h.plan['input']['checkpoint_grace_seconds'])
        (tmp_path/'stop.json').unlink()
        with pytest.raises(EvidenceError,match='disappeared'):check()


def test_failed_dispatch_parent_never_admits_successor_even_with_zero_exit(tmp_path):
    from run_sustained_pilot import parent_for
    dest=tmp_path/'stages/record';dest.mkdir(parents=True);write_json(dest/'dispatch-failure.json',{'failed':True})
    with pytest.raises(EvidenceError,match='cannot launch a successor'):
        parent_for({'parent_stage':'record'},[({'name':'record'},None)],tmp_path,None)


@pytest.mark.parametrize('installed',[True,False])
def test_completed_metadata_remains_binding_when_tensor_delivery_fails(tmp_path,installed):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        metadata=remote/'output/metadata.json';metadata.write_bytes(b'original synthetic metadata')
        versioned.export(t,'output',store,tmp_path/'completed-snapshot',int(time.time())+30)
        if not installed:
            obj=store/'objects'/file_hash(metadata);obj.unlink()
            for attempt in (store/'incoming').iterdir():
                selection=attempt/'download-selection.json'
                if selection.exists() and read_json(selection)['expected']['path']=='output/metadata.json':
                    # The selected tree inventory still binds bytes even if the
                    # completed transfer was interrupted before object install.
                    (attempt/'verified').unlink()
        metadata.write_bytes(b'changed synthetic metadata')
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        with pytest.raises(EvidenceError,match='contradicts preserved selection'):stop_and_retain(*args,**kw)
        assert not h.jobs[root]['finished'] and not(out/'failure-retention/retention.json').exists()


def test_terminal_contradiction_stays_fatal_on_restart(tmp_path,monkeypatch):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store);original=m.export
        def changed(*a,**k):
            terminal=remote/'jobs'/root/'exit.json';value=read_json(terminal);value['exit_code']=17;write_json(terminal,value)
            return original(*a,**k)
        monkeypatch.setattr(m,'export',changed)
        with pytest.raises(EvidenceError,match='terminal changed'):stop_and_retain(*args,**kw)
        assert read_json(out/'failure-retention/terminal.json')['exit_code']==0
        before=len(calls)
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):stop_and_retain(*args,**kw)
        assert len(calls)==before and not h.jobs[root]['finished']


@pytest.mark.parametrize('cleanup',[False,True])
def test_strict_supervision_failure_cannot_open_another_recovery_connection(tmp_path,monkeypatch,cleanup):
    import sustained_pilot_abort as abort
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store);seen=[]
        def denied(*a,**k):
            seen.append(True);error=EvidenceError('synthetic authentication failure')
            if cleanup:error.transport_cleanup_diagnostic={'exception_class':'PermissionError'}
            raise error
        monkeypatch.setattr(abort,'job_supervision',denied)
        with pytest.raises(EvidenceError,match='authentication failure'):stop_and_retain(*args,**kw)
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):stop_and_retain(*args,**kw)
        assert seen==[True] and not h.jobs[root]['finished']


def test_uncached_backup_allowance_is_not_replaced_by_logical_allowance(tmp_path):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    (remote/'output/payload').write_bytes(b'x'*(2*1024**2))
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        kw['failure_limits'].update(maximum_bytes=4*1024**2,maximum_uncached_bytes=1024**2)
        original=t.get;payload=[]
        def observed(name,*a,**k):
            if name=='output/payload':payload.append(True)
            return original(name,*a,**k)
        t.get=observed
        with pytest.raises(EvidenceError,match='uncached export bound'):stop_and_retain(*args,**kw)
        assert not payload and not h.jobs[root]['finished']


def test_completed_roots_adopt_after_interrupt_without_new_window(tmp_path,monkeypatch):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store);original=m.export;seen=[]
        def interrupted(transport,name,*a,**k):
            if name=='output':raise KeyboardInterrupt('synthetic between-root interruption')
            seen.append(name);return original(transport,name,*a,**k)
        monkeypatch.setattr(m,'export',interrupted)
        with pytest.raises(KeyboardInterrupt):stop_and_retain(*args,**kw)
        saved=(out/'failure-retention/intent.json').read_bytes()
        assert not(out/'failure-retention-failure.json').exists()
        monkeypatch.setattr(m,'export',original);result=stop_and_retain(*args,**kw)
        assert result['exit']['exit_code']==0 and (out/'failure-retention/intent.json').read_bytes()==saved
        assert len(seen)==1 and h.jobs[root]['finished']
