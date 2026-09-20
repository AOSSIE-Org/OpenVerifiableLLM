"""Actual detached stage, SSH subprocess exports and durable health behind local doubles."""
from pathlib import Path
from datetime import datetime,timezone
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_workload_stage as m
from workload_health import Health
from test_pod_job_client import fixture
from test_external_watchdog import intent as base_intent
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.supervision import Journal,rental_plan


def intent():
    w=base_intent();now=int(time.time())
    w['plan']=rental_plan({**w['plan']['input'],'now_epoch':now})
    w['creation_latest_epoch']=now+30
    w['payload']['terminateAfter']=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    w['baseline']['observed_epoch']=now;w['baseline']['http_clock']={'server_epoch':now,'request_started_epoch':now,'request_completed_epoch':now}
    return w


def staged(tmp_path):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path)
    output=remote/'output';output.mkdir();(output/'payload').write_bytes(b'complete explicitly synthetic output')
    value=read_json(job);value['export_roots']=[t.profile['remote_root']+'/output'];write_json(job,value);root=digest(value)
    return t,remote,calls,job,root,source,worker_root


def test_real_stage_exit_exports_then_cost_completion_without_reexecution(tmp_path):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);w=intent()
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,w,t.profile['pod_id'])
        value=m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64,sleep=lambda n:time.sleep(.05))
        assert value['exit']['exit_code']==0 and h.jobs[root]['finished'] and not h.complete
        assert len(value['exports'])==2
        before=len(calls)
        assert m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64)==value
        assert len(calls)==before
        final=tmp_path/'final';final.mkdir();write_json(final/'exit.json',value['exit'])
        h.finish(final,inventory(final,['exit.json']));assert h.write(tmp_path/'health.json')['complete']


def test_real_controller_stop_request_is_delivered_and_all_partial_outputs_preserved(tmp_path):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path)
    value=read_json(job);script=Path(value['argv'][1]);script.write_text('import time\nprint("preserved before stop",flush=True)\ntime.sleep(20)\n')
    from ovl_pipeline.canonical import file_hash
    value['required_files'][-1].update(bytes=script.stat().st_size,sha256=file_hash(script));write_json(job,value);root=digest(value)
    stop=tmp_path/'stop.json'
    def request_stop(n):
        write_json(stop,{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,'pod_id':t.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['test-controller-stop']})
        time.sleep(.05)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id'])
        result=m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',stop,'d'*64,sleep=request_stop)
        assert result['exit']['exit_code']<0 and h.jobs[root]['finished'] and not h.complete
        assert (tmp_path/'stage/stop-delivery.json').exists()
        records=Path(result['exports'][0]['directory']);assert (records/'request-stop').exists()
        assert 'preserved before stop' in (records/'stdout.log').read_text()


@pytest.mark.parametrize('kind',['production-record','full-replay'])
def test_unwired_production_stages_fail_before_any_remote_work(tmp_path,kind):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);value=read_json(job);value['kind']=kind;write_json(job,value)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id'])
        with pytest.raises(EvidenceError,match='checkpoint hooks'):
            m.run_stage(t,h,job,digest(value),source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64)
    assert not calls


def test_existing_stop_request_prevents_a_new_stage_launch(tmp_path):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);stop=tmp_path/'stop.json';write_json(stop,{'stop':'already requested'})
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id'])
        with pytest.raises(EvidenceError,match='before launch'):
            m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',stop,'d'*64)
    assert not calls


def test_dead_supervisor_is_adopted_stopped_exported_and_marked_abandoned(tmp_path):
    import signal
    import pod_job_worker as worker
    from ovl_pipeline.canonical import file_hash
    t,remote,calls,job,root,source,worker_root=staged(tmp_path)
    value=read_json(job);script=Path(value['argv'][1]);script.write_text('import time\nprint("retained orphan output",flush=True)\ntime.sleep(20)\n')
    value['required_files'][-1].update(bytes=script.stat().st_size,sha256=file_hash(script));write_json(job,value);root=digest(value)
    killed=[]
    def kill_runner(n):
        if not killed:
            selected=read_json(remote/'jobs'/root/'launch/receipt.json')['runner']
            worker.signal_owned(selected,signal.SIGKILL);killed.append(selected)
        time.sleep(.05)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id'])
        result=m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64,sleep=kill_runner)
        assert killed and result['exit']['state']=='ABANDONED' and result['exit']['exit_code']=='UNAVAILABLE'
        assert h.jobs[root]['finished'] and not h.complete
        final=tmp_path/'final';final.mkdir();write_json(final/'abandoned.json',result['exit'])
        h.finish(final,inventory(final,['abandoned.json']));assert h.write(tmp_path/'health.json')['complete']
        assert 'retained orphan output' in (Path(result['exports'][0]['directory'])/'stdout.log').read_text()


def test_expired_compute_deadline_allows_only_fenced_readonly_adoption_and_export(tmp_path,monkeypatch):
    from test_pod_job_worker import exited
    t,remote,calls,job,root,source,worker_root=staged(tmp_path)
    w=intent();value=read_json(job);value['deadline_epoch']=int(time.time())+3
    write_json(job,value);root=digest(value);out=tmp_path/'stage'
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,w,t.profile['pod_id'])
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        m.launch(t,job,root,source,worker_root,out/'launch',value['deadline_epoch'])
        assert exited(remote/'jobs'/root)['exit_code']==0
        while time.time()<=value['deadline_epoch']:time.sleep(.05)
        before=len(calls)
        def forbidden(*a,**k):raise AssertionError('adoption must not call launch')
        monkeypatch.setattr(m,'launch',forbidden)
        result=m.run_stage(t,h,job,root,source,worker_root,out,tmp_path/'health.json',tmp_path/'stop.json','d'*64)
        assert result['exit']['exit_code']==0 and len(result['exports'])==2
        assert not any(' start ' in c[-1] for c in calls[before:])


def test_expired_unfenced_compute_deadline_cannot_launch(tmp_path):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);w=intent()
    value=read_json(job);value['deadline_epoch']=int(time.time())+1
    write_json(job,value);root=digest(value)
    while time.time()<=value['deadline_epoch']:time.sleep(.05)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,w,t.profile['pod_id'])
        with pytest.raises(EvidenceError,match='bounded work'):
            m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64)
    assert not calls


@pytest.mark.parametrize('fault',['missing-final-status','noncanonical-activity','oversized-activity'])
def test_durable_terminal_exports_after_mutable_metadata_failure(tmp_path,fault):
    from test_pod_job_worker import exited
    t,remote,calls,job,root,source,worker_root=staged(tmp_path)
    value=read_json(job);value['environment']['OVL_ACTIVITY_FILE']=t.profile['remote_root']+'/output/activity.json'
    write_json(job,value);root=digest(value);out=tmp_path/'stage'
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id']);h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        m.launch(t,job,root,source,worker_root,out/'launch',int(time.time())+30)
        assert exited(remote/'jobs'/root)['exit_code']==0
        time.sleep(.1)
        if fault=='missing-final-status':(remote/'jobs'/root/'status.json').unlink()
        else:(remote/'output/activity.json').write_bytes(b'{ "not": "canonical" }\n' if fault=='noncanonical-activity' else b'x'*65537)
        result=m.run_stage(t,h,job,root,source,worker_root,out,tmp_path/'health.json',tmp_path/'stop.json','d'*64)
        assert result['exit']['exit_code']==0 and len(result['exports'])==2
        if fault!='missing-final-status':assert (Path(result['exports'][1]['directory'])/'activity.json').read_bytes()==(remote/'output/activity.json').read_bytes()


def test_health_recorded_but_unfenced_expired_stage_records_precise_refusal(tmp_path):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);w=intent()
    value=read_json(job);value['deadline_epoch']=int(time.time())+1;write_json(job,value);root=digest(value)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,w,t.profile['pod_id']);h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        while time.time()<=value['deadline_epoch']:time.sleep(.05)
        with pytest.raises(EvidenceError,match='bounded work'):
            m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64)
        assert not h.complete and not calls
        assert read_json(tmp_path/'stage/unlaunched-stage.json')['execution']=='NOT_STARTED_BY_THIS_COORDINATOR'


def test_partial_export_is_retained_and_one_fresh_retry_can_complete(tmp_path,monkeypatch):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);original=m.export_tree;failed=[]
    def fail_once(transport,name,output,*a,**k):
        if not failed:
            output.mkdir();(output/'retained.partial').write_bytes(b'actual selected prefix');failed.append(True)
            raise EvidenceError('injected interrupted export')
        return original(transport,name,output,*a,**k)
    monkeypatch.setattr(m,'export_tree',fail_once)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id']);args=(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64)
        with pytest.raises(EvidenceError,match='interrupted export'):m.run_stage(*args,sleep=lambda _:time.sleep(.05))
        result=m.run_stage(*args)
        assert result['exit']['exit_code']==0
        assert (tmp_path/'stage/export-000/retained.partial').read_bytes()==b'actual selected prefix'
        assert 'export-000-attempt-001' in result['exports'][0]['directory']
        assert len([c for c in calls if ' start ' in c[-1]])==1


def test_two_incomplete_export_attempts_are_not_deleted_or_retried_forever(tmp_path,monkeypatch):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path)
    def fail(transport,name,output,*a,**k):
        output.mkdir();(output/'retained.partial').write_bytes(b'preserved');raise EvidenceError('interrupted export')
    monkeypatch.setattr(m,'export_tree',fail)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id']);args=(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64)
        for _ in range(2):
            with pytest.raises(EvidenceError,match='interrupted export'):m.run_stage(*args,sleep=lambda _:time.sleep(.05))
        with pytest.raises(EvidenceError,match='retries exhausted'):m.run_stage(*args)
        for name in ('export-000','export-000-attempt-001'):assert (tmp_path/'stage'/name/'retained.partial').read_bytes()==b'preserved'
        assert not h.complete


@pytest.mark.parametrize('remaining',[30,200])
@pytest.mark.parametrize('adopting',[False,True])
def test_launch_control_allowance_never_changes_job_or_rental_deadline(tmp_path,monkeypatch,remaining,adopting):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);w=intent();now=int(time.time())
    value=read_json(job);value['deadline_epoch']=now+remaining;write_json(job,value);root=digest(value)
    out=tmp_path/'stage'
    if adopting:
        (out/'launch').mkdir(parents=True)
        write_json(out/'launch/launch-intent.json',{'schema':'ovl.offpod-job-launch-intent.v1','job_sha256':root,'worker_sha256':worker_root,'profile_sha256':digest(t.profile)})
    original_job=job.read_bytes();original_plan=digest(w['plan']);seen=[]
    class Selected(Exception):pass
    def intercept(*args,**kw):seen.append(args[-1]);raise Selected()
    monkeypatch.setattr(m,'reconcile_launch' if adopting else 'launch',intercept)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,w,t.profile['pod_id']);h.wall=lambda:now
        with pytest.raises(Selected):m.run_stage(t,h,job,root,source,worker_root,out,tmp_path/'health.json',tmp_path/'stop.json','d'*64)
        assert seen==[min(w['plan']['external_terminate_epoch'] if adopting else value['deadline_epoch'],now+120)]
    assert job.read_bytes()==original_job and digest(w['plan'])==original_plan and not calls


@pytest.mark.parametrize('stop_kind',['controller','graceful','job'])
def test_stop_during_uploads_forbids_unfenced_start(tmp_path,stop_kind):
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);w=intent();base=int(time.time());clock=[base]
    value=read_json(job);value['deadline_epoch']=base+(70 if stop_kind=='job' else 300)
    write_json(job,value);root=digest(value);stop=tmp_path/'stop.json';original=t.put;uploads=[]
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,w,t.profile['pod_id']);h.wall=lambda:clock[0]
        if stop_kind=='graceful':h.plan['request_checkpoint_epoch']=base+70
        original_plan=digest(h.plan);original_job=job.read_bytes();progress=h.progress
        def delayed(name,*args,**kwargs):
            result=original(name,*args,**kwargs);uploads.append(name);clock[0]+=40
            if len(uploads)==2 and stop_kind=='controller':
                write_json(stop,{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
                    'pod_id':h.pod,'observed_epoch':clock[0],'reasons':['synthetic stop during upload']})
            return result
        t.put=delayed
        with pytest.raises(EvidenceError,match='before launch fence'):
            m.run_stage(t,h,job,root,source,worker_root,tmp_path/'stage',tmp_path/'health.json',stop,'d'*64)
        assert len(uploads)==2 and not(tmp_path/'stage/launch/launch-intent.json').exists()
        assert h.progress==progress and not h.complete
        assert job.read_bytes()==original_job and digest(h.plan)==original_plan
    assert not any(' start ' in c[-1] for c in calls)


def test_pending_stop_is_delivered_before_failing_readonly_adoption(tmp_path,monkeypatch):
    from test_pod_job_worker import exited
    t,remote,calls,job,root,source,worker_root=staged(tmp_path);out=tmp_path/'stage'
    m.launch(t,job,root,source,worker_root,out/'launch',int(time.time())+30)
    assert exited(remote/'jobs'/root)['exit_code']==0
    stop=tmp_path/'stop.json';write_json(stop,{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
        'pod_id':t.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['synthetic pending stop']})
    def blocked(*args):
        assert (remote/'jobs'/root/'request-stop').read_bytes()==stop.read_bytes()
        raise EvidenceError('synthetic adoption read unavailable')
    monkeypatch.setattr(m,'reconcile_launch',blocked)
    with Journal(tmp_path/'journal').lease() as j:
        h=Health(j,intent(),t.profile['pod_id']);before=len(calls)
        with pytest.raises(EvidenceError,match='adoption read unavailable'):
            m.run_stage(t,h,job,root,source,worker_root,out,tmp_path/'health.json',stop,'d'*64)
        assert not h.complete and not any(' start ' in c[-1] for c in calls[before:])
