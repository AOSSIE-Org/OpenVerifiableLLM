"""Distinct authorized stop causes must preserve one immutable worker signal."""
import time
from pathlib import Path
import pytest
from ovl_pipeline.canonical import EvidenceError, digest, read_json, write_json
from ovl_pipeline.supervision import Journal
from workload_health import Health
from test_workload_stage import staged, intent
import run_workload_stage as stage
from sustained_pilot_abort import stop_and_retain


def test_dispatch_abort_then_controller_stop_retains_without_overwriting(tmp_path):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    value=read_json(job);program=Path(value['argv'][1])
    program.write_text('import time\nprint("synthetic live child",flush=True)\ntime.sleep(20)\n')
    from ovl_pipeline.canonical import file_hash
    value['required_files'][-1].update(bytes=program.stat().st_size,sha256=file_hash(program))
    write_json(job,value);root=digest(value)
    out=tmp_path/'stage';stop=tmp_path/'controller-stop.json' 
    with Journal(tmp_path/'journal').lease() as journal:
        health=Health(journal,intent(),t.profile['pod_id'])
        health.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':health.pod,'kind':'pilot'})
        stage.launch(t,job,root,worker,worker_root,out/'launch',int(time.time())+30)
        assert stage.job_supervision(t,root,worker_root,int(time.time())+10)['state'] in ('STARTING_UNVERIFIED','RUNNING_UNVERIFIED')
        write_json(stop,{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,'pod_id':health.pod,
                         'observed_epoch':int(time.time()),'reasons':['synthetic controller stop']})
        result=stop_and_retain(t,health,job,root,worker,worker_root,out,tmp_path/'health.json',stop,'d'*64,
                               sleep=lambda _:time.sleep(.05))
        assert result['exit']['state']=='EXITED' and health.jobs[root]['finished']
        assert len(result['exports'])==2
        assert (out/'dispatch-stop-delivery.json').exists() and result['exit']['exit_code']<0
        remote_stop=remote/'jobs'/root/'request-stop'
        before=(remote_stop.read_bytes(),remote_stop.stat().st_ino,remote_stop.stat().st_mtime_ns)
        assert stage.run_stage(t,health,job,root,worker,worker_root,out,tmp_path/'health.json',stop,'d'*64)==result
        assert before==(remote_stop.read_bytes(),remote_stop.stat().st_ino,remote_stop.stat().st_mtime_ns)
        assert len([c for c in calls if ' start ' in c[-1]])==1


@pytest.mark.parametrize('observed,expected',[(1200,1870),(1000,1770)])
def test_later_controller_reason_never_renews_first_graceful_stop(tmp_path,observed,expected):
    from run_production_stage import stop_window
    job={'deadline_epoch':4000,'stop_grace_seconds':30}
    plan={'request_checkpoint_epoch':1100,'input':{'now_epoch':900,'checkpoint_grace_seconds':1000}}
    first={'schema':'ovl.dispatcher-stop-request.v1','job_sha256':digest(job),'reason':'fixed graceful stop'}
    assert stop_window(tmp_path,first,job,plan,1100,200)==(first,1870)
    original=(tmp_path/'stop-intent.json').read_bytes()
    controller={'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,'pod_id':'synthetic-pod',
                'observed_epoch':observed,'reasons':['synthetic later observation']}
    assert stop_window(tmp_path,controller,job,plan,1300,200)==(first,expected)
    assert stop_window(tmp_path,controller,job,plan,1500,200)==(first,expected)
    assert (tmp_path/'stop-intent.json').read_bytes()==original
    assert read_json(tmp_path/'controller-stop-intent.json')['request']==controller
    with pytest.raises(EvidenceError):stop_window(tmp_path,{**controller,'observed_epoch':1400},job,plan,1500,200)
    with pytest.raises(EvidenceError,match='disappeared'):stop_window(tmp_path,first,job,plan,1500,200)


@pytest.mark.parametrize('bad',['foreign','malformed','oversized','symlink'])
def test_worker_stop_immutable_put_rejects_altered_or_unrelated_content(tmp_path,bad):
    from pod_job_client import worker_stop_request
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    dest=remote/'jobs'/root/'request-stop';dest.parent.mkdir(parents=True)
    expected=tmp_path/'worker-stop.json';write_json(expected,worker_stop_request(root))
    if bad=='foreign':write_json(dest,worker_stop_request('f'*64))
    elif bad=='malformed':dest.write_bytes(b'{broken')
    elif bad=='oversized':dest.write_bytes(b'x'*65537)
    else:
        target=tmp_path/'untouched';target.write_bytes(b'keep');dest.symlink_to(target)
    before=dest.read_bytes();inode=dest.lstat().st_ino
    with pytest.raises(EvidenceError):t.put('jobs/'+root+'/request-stop',expected,int(time.time())+10)
    assert dest.read_bytes()==before and dest.lstat().st_ino==inode


def test_sequential_identical_stop_causes_leave_remote_bytes_and_inode(tmp_path):
    from pod_job_client import worker_stop_request
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    first=tmp_path/'first-stop.json';second=tmp_path/'second-stop.json'
    write_json(first,worker_stop_request(root));write_json(second,worker_stop_request(root))
    name='jobs/'+root+'/request-stop';t.put(name,first,int(time.time())+10)
    dest=remote/name;before=(dest.read_bytes(),dest.stat().st_ino,dest.stat().st_mtime_ns)
    receipt=t.put(name,second,int(time.time())+10)
    assert receipt['bytes_sent']==len(before[0])
    assert before==(dest.read_bytes(),dest.stat().st_ino,dest.stat().st_mtime_ns)
