"""Real local finite jobs/transfers; rental and SSH endpoints explicitly substituted."""
from pathlib import Path
import sys
import time
from datetime import datetime,timezone
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_workload_coordinator as m
from test_workload_stage import staged
from test_rental_controller import intent as rental_intent
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json,file_hash
from ovl_pipeline.supervision import Journal,rental_plan,ControllerBusy


def fixture(tmp_path,*,second=False):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    rental=rental_intent();w=rental['watchdog_intent'];now=int(time.time())
    rental['quote']['observed_epoch']=rental['quote']['storage_observed_epoch']=now
    w['plan']=rental_plan({**w['plan']['input'],'now_epoch':now,'quote_sha256':digest(rental['quote'])})
    w['creation_latest_epoch']=now+30;w['baseline']['observed_epoch']=now
    w['baseline']['http_clock']={'server_epoch':now,'request_started_epoch':now,'request_completed_epoch':now}
    stamp=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    w['payload']['terminateAfter']=rental['payload']['terminateAfter']=stamp
    controller=tmp_path/'controller'
    with Journal(controller).lease() as j:
        j.append('creation-intent',rental);j.append('creation-observed',{'id':t.profile['pod_id']})
    heartbeat=tmp_path/'watchdog.json'
    write_json(heartbeat,{'schema':'ovl.external-watchdog-heartbeat.v1','intent_sha256':digest(w),
               'plan_sha256':digest(w['plan']),'external_terminate_epoch':w['plan']['external_terminate_epoch'],
               'observed_epoch':now,'state':'ARMED','pod_id':t.profile['pod_id']})
    plan={'schema':'ovl.finite-workload-plan.v3','rental_intent_sha256':digest(rental),'profile_sha256':digest(t.profile),
          'worker_sha256':worker_root,'timing':{'export_reserve_seconds':120,'transfer_floor_bytes_per_second':1024**2,
          'hash_floor_bytes_per_second':100*1024**2,'basis':'explicit synthetic fixture only; no live throughput measurement'},
          'uploads':[],'stages':[{'name':'first','job_path':job.name,'job_sha256':root,'maximum_export_bytes':1024**2}]}
    if second:
        another=read_json(job);another['environment']['LANG']='C';write_json(tmp_path/'job2.json',another)
        plan['stages'].append({'name':'second','job_path':'job2.json','job_sha256':digest(another),'maximum_export_bytes':1024**2})
    def run():return m.run(plan,digest(plan),rental,controller,heartbeat,t,tmp_path,worker,tmp_path/'coordinator',tmp_path/'health.json',sleep=lambda _:time.sleep(.05))
    return run,plan,rental,t,remote,calls,controller,heartbeat


def test_finite_two_stage_execution_and_completed_adoption_never_restarts(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True)
    result=run();assert result['outcome']=='EXITED_ZERO' and len(result['stages'])==2 and not result['unstarted_stages']
    assert read_json(tmp_path/'health.json')['complete'] is True
    before=len(calls);assert run()==result;assert len(calls)==before
    assert len([c for c in calls if ' start ' in c[-1]])==2
    exported=Path(result['stages'][0]['result_path']).parent/'export-000/files/stdout.log';exported.write_bytes(b'changed')
    with pytest.raises(EvidenceError):run()


def test_restart_after_real_stage_retains_original_job_and_completes(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    original=m.run_stage
    def crash(*a,**k):
        original(*a,**k);raise KeyboardInterrupt('explicit coordinator death after exported stage')
    monkeypatch.setattr(m,'run_stage',crash)
    with pytest.raises(KeyboardInterrupt):run()
    assert read_json(tmp_path/'health.json')['complete'] is False
    monkeypatch.setattr(m,'run_stage',original);before=len(calls)
    assert run()['outcome']=='EXITED_ZERO';assert len(calls)==before
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_death_before_final_health_and_new_stop_does_not_rewrite_saved_result(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    original=m.finalize
    def crash(*a,**k):raise KeyboardInterrupt('explicit coordinator death before final health')
    monkeypatch.setattr(m,'finalize',crash)
    with pytest.raises(KeyboardInterrupt):run()
    saved=read_json(tmp_path/'coordinator/final/result.json');assert saved['outcome']=='EXITED_ZERO'
    write_json(controller/'stop-request.json',{'explicit':'later controller stop'})
    monkeypatch.setattr(m,'finalize',original);before=len(calls)
    assert run()==saved and read_json(tmp_path/'health.json')['complete']
    assert len(calls)==before


@pytest.mark.parametrize('fault',['wrong-pod','wrong-intent','stop','stale-watchdog','changed-descriptor','production','duplicate-stage','changed-profile'])
def test_inconsistent_or_stopped_rental_refuses_before_remote_work(tmp_path,fault):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    if fault in ('wrong-pod','stop'):
        with Journal(controller).lease() as j:
            if fault=='wrong-pod':j.append('creation-observed',{'id':'foreign'})
            else:j.append('decision',{'action':'CHECKPOINT_AND_STOP'})
    elif fault=='wrong-intent':rental['quote']['storage_page_sha256']='a'*64
    elif fault=='stale-watchdog':
        h=read_json(heartbeat);h['observed_epoch']-=31;write_json(heartbeat,h)
    elif fault in ('changed-descriptor','production'):
        p=tmp_path/'job.json';job=read_json(p);job['kind']='production-record';write_json(p,job)
        if fault=='production':plan['stages'][0]['job_sha256']=digest(job)
    elif fault=='duplicate-stage':plan['stages'].append(plan['stages'][0])
    else:t.profile['host']='127.0.0.2'
    with pytest.raises(EvidenceError):run()
    assert not calls


def test_failed_stage_exports_and_stops_without_launching_remaining_plan(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True)
    job=read_json(tmp_path/'job.json');program=Path(job['argv'][1]);program.write_text('raise SystemExit(17)\n')
    job['required_files'][-1].update(bytes=program.stat().st_size,sha256=file_hash(program));write_json(tmp_path/'job.json',job)
    plan['stages'][0]['job_sha256']=digest(job)
    result=run();assert result['outcome']=='FAILED_OR_ABANDONED' and result['unstarted_stages']==['second']
    assert result['stages'][0]['exit']['exit_code']==17 and read_json(tmp_path/'health.json')['complete']
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_journaled_stop_without_stop_file_finalizes_retained_stage(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True);original=m.run_stage
    def journal_stop(*a,**k):
        result=original(*a,**k)
        with Journal(controller).lease() as j:j.append('decision',{'action':'CHECKPOINT_AND_STOP'})
        return result
    monkeypatch.setattr(m,'run_stage',journal_stop)
    result=run()
    assert result['outcome']=='STOPPED_AFTER_STAGE' and result['unstarted_stages']==['second']
    assert read_json(tmp_path/'health.json')['complete'] and not (controller/'stop-request.json').exists()
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_activity_cannot_overwrite_worker_control_metadata(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    path=tmp_path/'job.json';job=read_json(path);job['environment']['OVL_ACTIVITY_FILE']=t.profile['remote_root']+'/jobs/other/exit.json'
    write_json(path,job);plan['stages'][0]['job_sha256']=digest(job)
    with pytest.raises(EvidenceError,match='worker-owned'):run()
    assert not calls


def test_duplicate_coordinator_lease_does_not_touch_remote_job(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    out=tmp_path/'coordinator';out.mkdir()
    with Journal(out/'health-journal').lease():
        with pytest.raises(ControllerBusy):run()
    assert not calls


def test_complete_input_upload_is_pinned_and_not_retransmitted_after_restart(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    source=tmp_path/'selected.bin';source.write_bytes(b'x'*1048577)
    plan['uploads']=[{'path':source.name,'remote_path':'inputs/selected.bin','bytes':source.stat().st_size,'sha256':file_hash(source)}]
    job=read_json(tmp_path/'job.json');job['required_files'].append({'path':t.profile['remote_root']+'/inputs/selected.bin','bytes':source.stat().st_size,'sha256':file_hash(source)})
    write_json(tmp_path/'job.json',job);plan['stages'][0]['job_sha256']=digest(job)
    def crash(*a,**k):raise KeyboardInterrupt('explicit crash after all setup transfers')
    monkeypatch.setattr(m,'run_stage',crash)
    with pytest.raises(KeyboardInterrupt):run()
    assert (remote/'inputs/selected.bin').read_bytes()==source.read_bytes()
    original_put=t.put
    def put(name,*a,**k):
        assert name!='inputs/selected.bin';return original_put(name,*a,**k)
    t.put=put
    with pytest.raises(KeyboardInterrupt,match='all setup transfers'):run()


@pytest.mark.parametrize('fault',['changed-bytes','control-path','duplicate-target'])
def test_unselected_upload_refused_before_transfer(tmp_path,fault):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    source=tmp_path/'selected.bin';source.write_bytes(b'abc')
    item={'path':source.name,'remote_path':'inputs/selected.bin','bytes':3,'sha256':file_hash(source)}
    plan['uploads']=[item]
    if fault=='changed-bytes':source.write_bytes(b'bad')
    elif fault=='control-path':item['remote_path']='jobs/selected/job.json'
    else:plan['uploads'].append(item)
    with pytest.raises(EvidenceError):run()
    assert not calls


def test_completion_health_is_restored_after_finish_write_crash(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    original=m.Health.write;failed=[]
    def crash(self,*a,**k):
        if self.complete and not failed:failed.append(True);raise OSError('explicit failed completion write')
        return original(self,*a,**k)
    monkeypatch.setattr(m.Health,'write',crash)
    with pytest.raises(OSError):run()
    assert read_json(tmp_path/'health.json')['complete'] is False
    before=len(calls);assert run()['outcome']=='EXITED_ZERO'
    assert read_json(tmp_path/'health.json')['complete'] and len(calls)==before


def test_copied_coordinator_cannot_return_unverified_moved_result(tmp_path):
    import shutil
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path);run()
    original=tmp_path/'coordinator';moved=tmp_path/'moved';shutil.copytree(original,moved)
    result=read_json(moved/'final/result.json');result['outcome']='forged';write_json(moved/'final/result.json',result)
    with pytest.raises(EvidenceError,match='directory moved'):
        m.run(plan,digest(plan),rental,controller,heartbeat,t,tmp_path,Path(m.__file__).with_name('pod_job_worker.py'),moved,tmp_path/'health2.json')


@pytest.mark.parametrize('damage',['long-stop','late-deadline','unaffordable-export','inputs-export','tools-export','foreign-activity','bad-env','unhashed-upload','huge-retry','nonisolated-setup'])
def test_static_admission_refuses_before_any_remote_work(tmp_path,damage):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    path=tmp_path/'job.json';job=read_json(path)
    if damage=='long-stop':job['stop_grace_seconds']=300
    elif damage=='late-deadline':job['deadline_epoch']=rental['watchdog_intent']['plan']['request_checkpoint_epoch']-1
    elif damage=='unaffordable-export':plan['stages'][0]['maximum_export_bytes']=2**40
    elif damage in ('inputs-export','tools-export'):job['export_roots']=[t.profile['remote_root']+'/'+damage.split('-')[0]]
    elif damage=='foreign-activity':job['environment']['OVL_ACTIVITY_FILE']='/outside/activity'
    elif damage=='bad-env':job['environment']['INJECTED_KEY']='explicit noncredential'
    elif damage=='nonisolated-setup':job['kind']='setup'
    else:
        source=tmp_path/'selected.bin';source.write_bytes(b'abc')
        plan['uploads']=[{'path':source.name,'remote_path':'inputs/selected.bin','bytes':3,'sha256':file_hash(source)}]
        if damage=='huge-retry':plan['uploads'][0]['bytes']=181*plan['timing']['transfer_floor_bytes_per_second']
    write_json(path,job);plan['stages'][0]['job_sha256']=digest(job)
    with pytest.raises(EvidenceError):run()
    assert not calls


def test_large_failed_transfer_retry_keeps_highwater_and_fixed_attempt_bound(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    source=tmp_path/'large.bin';source.write_bytes(b'x'*(10*1024**2))
    upload={'path':source.name,'remote_path':'inputs/large.bin','bytes':source.stat().st_size,'sha256':file_hash(source)}
    plan['uploads']=[upload]
    job=read_json(tmp_path/'job.json');job['required_files'].append({'path':t.profile['remote_root']+'/'+upload['remote_path'],'bytes':upload['bytes'],'sha256':upload['sha256']})
    write_json(tmp_path/'job.json',job);plan['stages'][0]['job_sha256']=digest(job)
    attempts=[]
    def interrupted(name,path,deadline,*,progress):
        now=int(time.time());attempts.append(deadline)
        assert now<deadline<=now+40
        for n in (1024**2,9*1024**2):progress({'bytes_sent':n,'bytes_received':0})
        raise OSError('explicit transfer loss at ninety percent')
    monkeypatch.setattr(t,'put',interrupted)
    with pytest.raises(OSError):run()
    events=Journal(tmp_path/'coordinator/health-journal')._read()
    before=[e for e in events if e['body'].get('kind')=='bytes']
    with pytest.raises(OSError):run()
    after=[e for e in Journal(tmp_path/'coordinator/health-journal')._read() if e['body'].get('kind')=='bytes']
    assert before==after and len(attempts)==2
    assert not list((tmp_path/'coordinator/uploads').glob('*.json'))
    assert list((tmp_path/'coordinator/controller-observations').glob('*.json'))


def test_all_stage_deadlines_are_statically_bounded_before_upload(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path)
    job=read_json(tmp_path/'job.json');job['deadline_epoch']=rental['watchdog_intent']['plan']['input']['now_epoch']+1501
    write_json(tmp_path/'job.json',job);plan['stages'][0]['job_sha256']=digest(job)
    with pytest.raises(EvidenceError,match='window'):run()
    assert not calls


def test_oversized_output_is_retained_but_cannot_advance_or_complete(tmp_path):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True)
    plan['stages'][0]['maximum_export_bytes']=1
    with pytest.raises(EvidenceError,match='exceeded selected export budget'):run()
    stage=tmp_path/'coordinator/stages/first/stage-result.json'
    assert read_json(stage)['exit']['exit_code']==0
    assert not read_json(tmp_path/'health.json')['complete']
    before=len(calls)
    with pytest.raises(EvidenceError,match='exceeded selected export budget'):run()
    assert len(calls)==before
