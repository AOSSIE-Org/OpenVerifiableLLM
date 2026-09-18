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
