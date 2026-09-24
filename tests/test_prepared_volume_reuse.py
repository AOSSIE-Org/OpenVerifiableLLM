"""Synthetic immutable-input reuse and tenant-quota regression cases."""
import os
from pathlib import Path
import time
from types import SimpleNamespace
import pytest
import pod_fetch_prepared as fetcher
import pod_job_worker as worker
from test_prepared_input_download import fixture
from ovl_pipeline.canonical import file_hash,read_json


def cached(tmp_path,monkeypatch):
    plan,v,bodies,out,report,opener=fixture(tmp_path,monkeypatch)
    source=tmp_path/'closed-inputs';source.mkdir()
    for name,data in bodies.items():
        p=source/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(data);p.chmod(0o444)
    return plan,v,bodies,out,report,opener,source


def test_full_hash_reuse_without_network_or_new_payload(tmp_path,monkeypatch):
    plan,v,bodies,out,report,opener,source=cached(tmp_path,monkeypatch)
    result=fetcher.fetch(plan,file_hash(plan),out,report,int(time.time())+60,opener=opener,reuse=source)
    assert not opener.calls and result['result']=='PASS'
    assert result['reused_bytes']==result['bytes']==sum(map(len,bodies.values())) and result['downloaded_bytes']==0
    assert result['raw_reconstruction']==result['training_replay']=='NOT_RUN'
    for name,data in bodies.items():
        assert (out/name).read_bytes()==data and os.path.samefile(source/name,out/name)
    assert read_json(report.parent/'activity.json')['schema']=='ovl.public-input-cache-read.v1'


@pytest.mark.parametrize('damage',['hash','missing','writable','symlink','parent-link','directory','fifo'])
def test_bad_selected_cache_never_falls_back_or_reports_success(tmp_path,monkeypatch,damage):
    plan,v,bodies,out,report,opener,source=cached(tmp_path,monkeypatch)
    p=source/fetcher.FILES[0]
    if damage=='hash':p.chmod(0o644);p.write_bytes(b'X'*p.stat().st_size);p.chmod(0o444)
    elif damage=='writable':p.chmod(0o644)
    elif damage=='parent-link':
        parent=p.parent;other=parent.with_name('displaced');parent.rename(other);parent.symlink_to(other,target_is_directory=True)
    else:
        p.unlink()
        if damage=='symlink':p.symlink_to(source/fetcher.FILES[1])
        elif damage=='directory':p.mkdir()
        elif damage=='fifo':os.mkfifo(p)
    with pytest.raises((ValueError,OSError)):
        fetcher.fetch(plan,file_hash(plan),out,report,int(time.time())+60,opener=opener,reuse=source)
    assert not opener.calls and not report.exists()


def test_reuse_deadline_during_hash_has_no_success(tmp_path,monkeypatch):
    plan,v,bodies,out,report,opener,source=cached(tmp_path,monkeypatch)
    now=[100.0]
    def clock():now[0]+=1;return now[0]
    with pytest.raises(TimeoutError):fetcher.fetch(plan,file_hash(plan),out,report,110,opener=opener,reuse=source,wall=clock)
    assert not report.exists() and not opener.calls


def test_link_time_source_replacement_is_rejected(tmp_path,monkeypatch):
    plan,v,bodies,out,report,opener,source=cached(tmp_path,monkeypatch)
    real_link=os.link
    def swap(src,dst,**kw):
        src=Path(src);src.unlink();src.write_bytes(b'X'*len(bodies[fetcher.FILES[0]]));src.chmod(0o444)
        real_link(src,dst,**kw)
    monkeypatch.setattr(fetcher.os,'link',swap)
    with pytest.raises(ValueError,match='identity changed'):fetcher.fetch(plan,file_hash(plan),out,report,int(time.time())+60,opener=opener,reuse=source)
    assert not report.exists() and not opener.calls


def test_tenant_quota_overrules_large_filesystem_pool(tmp_path,monkeypatch):
    (tmp_path/'data').write_bytes(b'x'*9000);job=tmp_path/'job';job.mkdir()
    used=worker.volume_usage(tmp_path,lambda:None)
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**15))
    env={'OVL_VOLUME_ROOT':str(tmp_path),'OVL_VOLUME_QUOTA_BYTES':str(used+8192)}
    guard=worker.VolumeGuard(env,job,lambda:None)
    assert guard.available()==8192
    env['OVL_VOLUME_QUOTA_BYTES']=str(used-1)
    assert worker.VolumeGuard(env,job,lambda:None).available()==0


def test_census_counts_hardlinks_once_sparse_logical_and_no_symlink_follow(tmp_path):
    p=tmp_path/'data';p.write_bytes(b'x'*9000);before=worker.volume_usage(tmp_path,lambda:None)
    os.link(p,tmp_path/'link');assert worker.volume_usage(tmp_path,lambda:None)==before
    (tmp_path/'outside-link').symlink_to('/not-traversed')
    with (tmp_path/'sparse').open('wb') as f:f.truncate(1024**2)
    assert worker.volume_usage(tmp_path,lambda:None)>=before+1024**2


def test_quota_refresh_observes_growth_and_checks_original_deadline(tmp_path,monkeypatch):
    now=[0];job=tmp_path/'job';job.mkdir()
    env={'OVL_VOLUME_ROOT':str(tmp_path),'OVL_VOLUME_QUOTA_BYTES':'10000000'}
    guard=worker.VolumeGuard(env,job,lambda:None,clock=lambda:now[0]);first=guard.available()
    (tmp_path/'data').write_bytes(b'x'*100000);now[0]=31
    assert guard.available()<first-90000
    def expired():raise worker.Refusal('original deadline expired')
    with pytest.raises(worker.Refusal,match='deadline'):worker.volume_usage(tmp_path,expired)


@pytest.mark.parametrize('env',[
 {'OVL_VOLUME_ROOT':'/volume'}, {'OVL_VOLUME_QUOTA_BYTES':'100'},
 {'OVL_VOLUME_ROOT':'/','OVL_VOLUME_QUOTA_BYTES':'100'},
 {'OVL_VOLUME_ROOT':'/volume/../other','OVL_VOLUME_QUOTA_BYTES':'100'},
 {'OVL_VOLUME_ROOT':'/volume','OVL_VOLUME_QUOTA_BYTES':'0'},
 {'OVL_VOLUME_ROOT':'/volume','OVL_VOLUME_QUOTA_BYTES':'nan'},
])
def test_invalid_quota_selection_fails_closed(env):
    with pytest.raises(worker.Refusal):worker.volume_selection(env)


def test_actual_worker_refuses_quota_before_spawning_child(tmp_path):
    from test_pod_job_worker import fixture,start,exited
    from ovl_pipeline.canonical import write_json,digest
    job,value,_,code_hash=fixture(tmp_path,'from pathlib import Path\nPath("must-not-run").write_text("x")\n')
    value['environment'].update(OVL_VOLUME_ROOT=str(tmp_path),OVL_VOLUME_QUOTA_BYTES='1')
    write_json(job/'job.json',value)
    assert start(job,digest(value),code_hash).returncode==0
    assert exited(job)['exit_code']==125 and not (tmp_path/'must-not-run').exists()
    assert not (job/'launch/child.json').exists()


def test_qualified_quota_pins_cannot_disappear_or_drift(tmp_path):
    from production_run_inputs import qualified_volume
    from ovl_pipeline.canonical import write_json,digest,EvidenceError
    env={'OVL_VOLUME_ROOT':'/selected-volume','OVL_VOLUME_QUOTA_BYTES':'224000000000'}
    stages=[]
    for n in range(2):
        job={'environment':env};name=f'job{n}.json';write_json(tmp_path/name,job)
        stages.append({'template_path':name,'template_sha256':digest(job)})
    plan={'stages':stages};assert qualified_volume(plan,tmp_path)==env
    job={'environment':{}};write_json(tmp_path/'job1.json',job);stages[1]['template_sha256']=digest(job)
    with pytest.raises(EvidenceError,match='different volume'):qualified_volume(plan,tmp_path)
    stages[1]['template_sha256']='f'*64
    with pytest.raises(EvidenceError,match='changed'):qualified_volume(plan,tmp_path)


def test_cache_progress_is_bounded_not_export_credit_and_cannot_switch_origin(tmp_path):
    from test_sustained_health import health,transfer
    from test_workload_health import Clock,JOB,SELECTION
    from test_external_watchdog import NOW
    from ovl_pipeline.supervision import Journal
    from ovl_pipeline.canonical import EvidenceError
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job({**SELECTION,'kind':'setup'});c.advance(10)
        value={**transfer(1024**2),'schema':'ovl.public-input-cache-read.v1'}
        assert h.activity(JOB,value);assert h.progress==NOW+10 and h.exported==NOW
        assert not h.activity(JOB,value)
        with pytest.raises(EvidenceError,match='process changed'):h.activity(JOB,transfer(2*1024**2))
    with Journal(path).lease() as j:
        h=health(j,c);assert not h.activity(JOB,value)
        c.advance(10);assert h.activity(JOB,{**value,'received_bytes':2*1024**2})


def run_selected_direct(tmp_path,monkeypatch,guard_class,*,request_stop=False):
    import threading
    from test_pod_job_worker import fixture
    from ovl_pipeline.canonical import write_json,digest
    job,v,_,worker_hash=fixture(tmp_path,'import time\ntime.sleep(20)\n',seconds=10)
    v['environment'].update(OVL_VOLUME_ROOT=str(tmp_path),OVL_VOLUME_QUOTA_BYTES='100000000')
    v['stop_grace_seconds']=2;write_json(job/'job.json',v);root=digest(v)
    (job/'launch').mkdir();write_json(job/'launch/intent.json',{'schema':'ovl.pod-job-launch-intent.v1','job_sha256':root,'worker_sha256':worker_hash})
    monkeypatch.setattr(worker,'VolumeGuard',guard_class)
    result=[];errors=[]
    def invoke():
        try:result.append(worker.run(job,root,worker_hash))
        except BaseException as e:errors.append(e)
    t=threading.Thread(target=invoke);t.start();end=time.monotonic()+5
    try:
        while not (job/'launch/child.json').exists() and time.monotonic()<end:time.sleep(.01)
        assert (job/'launch/child.json').exists()
        sent=time.monotonic()
        if request_stop:(job/'request-stop').write_text('synthetic stop')
        t.join(7);assert not t.is_alive() and not errors
        return job,result[0],time.monotonic()-sent
    finally:
        if t.is_alive():
            (job/'request-stop').write_text('test cleanup');t.join(12)


def test_live_census_does_not_bypass_operator_stop_grace(tmp_path,monkeypatch):
    base=worker.VolumeGuard
    class Refresh(base):
        def available(self):self.checked_at=None;return super().available()
    job,result,elapsed=run_selected_direct(tmp_path,monkeypatch,Refresh,request_stop=True)
    assert result['exit_code']<0 and elapsed>=2
    assert not (job/'failure.json').exists()
    assert read_json(job/'status.json')['stop_reason']=='operator-stop'


def test_live_quota_low_observation_stops_child_and_retains_terminal(tmp_path,monkeypatch):
    base=worker.VolumeGuard
    class LowAfterSpawn(base):
        def available(self):
            if (self.directory/'launch/child.json').exists():return 0
            return super().available()
    job,result,_=run_selected_direct(tmp_path,monkeypatch,LowAfterSpawn)
    assert result['exit_code']<0 and not (job/'failure.json').exists()
    assert read_json(job/'status.json')['stop_reason']=='storage-bound'
