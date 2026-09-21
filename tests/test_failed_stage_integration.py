"""Full dispatcher failure/backup/finalization with real local owned processes."""
from pathlib import Path
import time
import pytest

from ovl_pipeline.canonical import digest,read_json,write_json
from pod_transfer import TransientTransportError,RangeRecoveryExhausted
from test_sustained_pilot_dispatch import fixture


def select_pilot(plan,tmp_path,t):
    first=plan['stages'][0];path=tmp_path/first['template_path'];job=read_json(path)
    job['kind']='pilot';job['environment']['OVL_ACTIVITY_FILE']=t.profile['remote_root']+'/output/activity.json'
    write_json(path,job);binding={'schema':'ovl.pilot-record-parent-binding.v1','recipe_sha256':'a'*64,
        'kernel_sha256':'b'*64,'stream_sha256':'c'*64,'code_root':'d'*64}
    first.update(template_sha256=digest(job),parent_binding=binding,
        validation_binding={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':'c'*64,'documents':1},
        retention={'schema':'ovl.pilot-initial-retention.v1','mode':'record','output_root':t.profile['remote_root']+'/output',
                   'phase':'wikipedia','maximum_initial_bytes':1024**2})


def test_exhausted_terminal_export_completes_failed_retention_without_successor(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True)
    select_pilot(plan,tmp_path,t)
    original=t.get;failed=[]
    def fail_once(name,destination,expected,deadline,**kwargs):
        if name=='output/payload' and not failed:
            failed.append(True);destination.with_name(destination.name+'.partial').write_bytes((remote/name).read_bytes()[:5])
            cause=TransientTransportError('synthetic exhausted payload injection');cause.transfer_counts={'bytes_sent':0,'bytes_received':5}
            raise RangeRecoveryExhausted('synthetic exhausted finite recovery') from cause
        return original(name,destination,expected,deadline,**kwargs)
    t.get=fail_once;result=run()
    assert failed and result['outcome']=='DISPATCH_FAILED_AFTER_RETENTION'
    assert result['failure']['error_type']=='RangeRecoveryExhausted' and result['unstarted_stages']==['second']
    assert result['stages'][0]['exit']['exit_code']==0 and result['production_acceptance']=='NOT_RUN'
    assert read_json(tmp_path/'health.json')['complete'] is True
    out=tmp_path/'sustained/stages/first'
    assert not(out/'stage-result.json').exists()
    proof=read_json(out/'failure-retention/retention.json');assert proof['qualification']=='FAILED'
    before=len(calls);assert run()==result and len(calls)==before
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_enclosing_abort_adopts_only_failed_backup_after_interruption(tmp_path,monkeypatch):
    from test_production_run_coordinator import configured
    from production_run_coordinator import Run,restore_phase_bindings
    import failed_stage_retention as backup
    from ovl_pipeline.canonical import EvidenceError
    factory,plan,initial,selection,t,remote,calls=configured(tmp_path,monkeypatch)
    select_pilot(plan,tmp_path,t);selection['phases']['qualification']=digest(plan)
    original=t.get;exhausted=[]
    def fail_once(name,destination,expected,deadline,**kwargs):
        if name=='output/payload' and not exhausted:
            exhausted.append(True);destination.with_name(destination.name+'.partial').write_bytes((remote/name).read_bytes()[:5])
            cause=TransientTransportError('synthetic range interruption');cause.transfer_counts={'bytes_sent':0,'bytes_received':5}
            raise RangeRecoveryExhausted('synthetic exhausted finite recovery') from cause
        return original(name,destination,expected,deadline,**kwargs)
    t.get=fail_once;original_export=backup.export
    def interrupted(transport,name,*a,**k):
        if name=='output':raise KeyboardInterrupt('synthetic between-root process interruption')
        return original_export(transport,name,*a,**k)
    monkeypatch.setattr(backup,'export',interrupted)
    with pytest.raises(KeyboardInterrupt):
        with factory() as owner:
            owner.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
    base=tmp_path/'qualification/stages/first/failure-retention';saved=(base/'intent.json').read_bytes()
    assert (base/'export-000/returned.json').exists() and not(base/'export-001').exists()
    assert (tmp_path/'coordinator/failure/reason.json').exists()
    bindings={};downloads={};restore_phase_bindings(plan,digest(plan),tmp_path/'qualification',bindings,downloads)
    rental=read_json(tmp_path/'full-controller/creation-intent.json') if (tmp_path/'full-controller/creation-intent.json').exists() else owner.rental
    monkeypatch.setattr(backup,'export',original_export)
    resumed=Run(selection,digest(selection),rental,t,Path('scripts/pod_job_worker.py'),tmp_path/'full-controller',
                tmp_path/'watchdog.json',tmp_path/'coordinator',tmp_path/'health.json',bindings,downloads,sleep=lambda _:time.sleep(.01))
    with pytest.raises(EvidenceError,match='prior failed run retained and stopped'):
        with resumed:raise AssertionError('failed enclosing run cannot return admission authority')
    assert (base/'intent.json').read_bytes()==saved and read_json(tmp_path/'health.json')['complete']
    assert read_json(base/'retention.json')['qualification']=='FAILED'
    assert len([c for c in calls if ' start ' in c[-1]])==1
    assert not(tmp_path/'coordinator/qualification-parents.json').exists()
