"""Actual detached Linux processes; no paid endpoint, CUDA or training credit."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_job_worker as m
from ovl_pipeline.canonical import digest,file_hash,read_json,write_json


WORKER=Path(m.__file__).resolve()


def fixture(tmp_path,code,seconds=20):
    job=tmp_path/'job';job.mkdir();script=tmp_path/'program.py';script.write_text(code)
    executable=Path(sys.executable).resolve()
    value={'schema':'ovl.pod-job.v1','kind':'pilot','argv':[str(executable),str(script)],'cwd':str(tmp_path),
           'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},'deadline_epoch':int(time.time())+seconds,
           'stop_grace_seconds':1,'minimum_free_bytes':1,'required_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in [executable,script]],
           'export_roots':[str(job)]}
    write_json(job/'job.json',value)
    return job,value,digest(value),file_hash(WORKER)


def start(job,root,worker):
    return subprocess.run([sys.executable,str(WORKER),'start',str(job),root,worker],capture_output=True,timeout=5)


def exited(job,timeout=8):
    end=time.monotonic()+timeout
    while not(job/'exit.json').exists() and time.monotonic()<end:time.sleep(.05)
    if not(job/'exit.json').exists():
        # Test owns these subprocesses. Preserve files, kill only recorded process
        # groups to avoid a failing test leaving compute alive on the local host.
        for name,field in [('child.json','process'),('receipt.json','runner')]:
            p=job/'launch'/name
            if p.exists():
                try:m.signal_owned(read_json(p)[field],signal.SIGKILL)
                except (ProcessLookupError,FileNotFoundError):pass
        pytest.fail('owned job did not produce bounded terminal receipt')
    return read_json(job/'exit.json')


def test_job_survives_launcher_exit_and_drops_inherited_secrets(tmp_path,monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY','explicit-synthetic-not-a-secret')
    job,v,root,worker=fixture(tmp_path,'import os,time\ntime.sleep(.2)\nprint("KEY_PRESENT", "RUNPOD_API_KEY" in os.environ)\n')
    result=start(job,root,worker);assert result.returncode==0
    receipt=json.loads(result.stdout);assert receipt['job_sha256']==root
    assert exited(job)['exit_code']==0 and (job/'stdout.log').read_text().strip()=='KEY_PRESENT False'
    assert read_json(job/'launch/child.json')['process']['pid']!=receipt['runner']['pid']
    assert read_json(job/'input-progress.json')['verified_bytes']==sum(f['bytes'] for f in v['required_files'])


@pytest.mark.parametrize('kind',['start-again','run-again','lost-receipt'])
def test_uncertain_or_completed_launch_never_reexecutes(tmp_path,kind):
    job,v,root,worker=fixture(tmp_path,'from pathlib import Path\np=Path("executions");p.write_text(p.read_text()+"x" if p.exists() else "x")\n')
    assert start(job,root,worker).returncode==0;assert exited(job)['exit_code']==0
    if kind=='lost-receipt':(job/'launch/receipt.json').rename(job/'launch/preserved-lost-receipt.json')
    operation='run' if kind=='run-again' else 'start'
    result=subprocess.run([sys.executable,str(WORKER),operation,str(job),root,worker],capture_output=True,timeout=5)
    assert result.returncode!=0 and (tmp_path/'executions').read_text()=='x'
    assert not(job/'failure.json').exists()


@pytest.mark.parametrize('damage',['job-hash','worker-hash','credential-env','missing-required-executable'])
def test_unselected_commands_or_environment_never_start(tmp_path,damage):
    job,v,root,worker=fixture(tmp_path,'raise SystemExit(0)\n')
    if damage=='job-hash':root='a'*64
    elif damage=='worker-hash':worker='b'*64
    else:
        if damage=='credential-env':v['environment']['HF_TOKEN']='synthetic-test-only'
        else:v['required_files']=v['required_files'][1:]
        write_json(job/'job.json',v);root=digest(v)
    assert start(job,root,worker).returncode!=0 and not(job/'launch').exists()


def test_required_file_change_produces_terminal_failure_without_child(tmp_path):
    job,v,root,worker=fixture(tmp_path,'raise SystemExit(0)\n');Path(v['argv'][1]).write_text('changed')
    assert start(job,root,worker).returncode==0
    status=exited(job);assert status['exit_code']==125
    assert read_json(job/'failure.json')['retry'].startswith('FORBIDDEN')
    assert not(job/'launch/child.json').exists()


@pytest.mark.parametrize('reason',['deadline','request-stop','log-bound'])
def test_deadline_stop_or_storage_bound_ends_owned_process_group(tmp_path,reason):
    code='import time\ntime.sleep(20)\n'
    if reason=='log-bound':code='import os,time\nos.write(1,b"x"*(17*1024*1024))\ntime.sleep(20)\n'
    job,v,root,worker=fixture(tmp_path,code,seconds=2 if reason=='deadline' else 20)
    before=time.monotonic();assert start(job,root,worker).returncode==0
    if reason=='request-stop':(job/'request-stop').write_text('owned stop request')
    status=exited(job);assert status['exit_code']<0 and time.monotonic()-before<6
    expected={'deadline':'job-deadline','request-stop':'operator-stop','log-bound':'storage-bound'}[reason]
    # exit.json is durable before the final mutable status write. CI observed
    # the prior RUNNING status in that window; wait for the status whose fields
    # this assertion actually checks. The process-stop bound above is unchanged.
    end=time.monotonic()+2;final=None
    while time.monotonic()<end:
        if (job/'status.json').exists():
            final=read_json(job/'status.json')
            if final['state']=='EXITED':break
        time.sleep(.01)
    assert final is not None and final['state']=='EXITED'
    assert final['job_sha256']==root and final['stop_reason']==expected


def test_pid_reuse_cannot_signal_another_process(monkeypatch):
    selected={'pid':12345,'start_ticks':100,'process_group':12345};signals=[]
    monkeypatch.setattr(m,'process_identity',lambda p:{**selected,'start_ticks':101})
    monkeypatch.setattr(m.os,'killpg',lambda *a:signals.append(a))
    with pytest.raises(m.Refusal,match='identity changed'):m.signal_owned(selected,signal.SIGTERM)
    assert signals==[]


def test_exited_leader_cannot_leave_a_background_group_writer(tmp_path):
    code='import subprocess,sys\nsubprocess.Popen([sys.executable,"-c","import time;from pathlib import Path;time.sleep(2);Path(\\\"late-write\\\").write_text(\\\"orphan\\\")"])\n'
    job,v,root,worker=fixture(tmp_path,code)
    assert start(job,root,worker).returncode==0 and exited(job)['exit_code']==0
    time.sleep(2.2)
    assert not(tmp_path/'late-write').exists()


def test_duplicate_live_runner_does_not_write_failure_into_active_job(tmp_path):
    job,v,root,worker=fixture(tmp_path,'import time\ntime.sleep(1.5)\n')
    assert start(job,root,worker).returncode==0
    end=time.monotonic()+5
    while not(job/'launch/child.json').exists() and time.monotonic()<end:time.sleep(.05)
    result=subprocess.run([sys.executable,str(WORKER),'run',str(job),root,worker],capture_output=True,timeout=5)
    assert result.returncode!=0 and exited(job)['exit_code']==0 and not(job/'failure.json').exists()


def test_absent_supervisor_can_stop_recorded_orphan_without_fabricating_exit_code(tmp_path):
    job,v,root,worker=fixture(tmp_path,'import time\ntime.sleep(20)\n')
    assert start(job,root,worker).returncode==0
    end=time.monotonic()+5
    while not(job/'launch/child.json').exists() and time.monotonic()<end:time.sleep(.05)
    child=read_json(job/'launch/child.json')['process'];runner=read_json(job/'launch/receipt.json')['runner']
    try:
        m.signal_owned(runner,signal.SIGKILL)
        while m.alive(runner) and time.monotonic()<end:time.sleep(.05)
        state=m.supervision(job,root,worker)
        assert state['state']=='SUPERVISOR_ABSENT' and state['child_alive']
        terminal=m.abandon(job,root,worker)
        assert terminal['state']=='ABANDONED' and terminal['exit_code']=='UNAVAILABLE'
        assert not m.alive(child) and not(job/'exit.json').exists()
        assert m.supervision(job,root,worker)['state']=='ABANDONED'
        assert start(job,root,worker).returncode!=0
    finally:
        if m.alive(child):m.signal_owned(child,signal.SIGKILL)
        if m.alive(runner):m.signal_owned(runner,signal.SIGKILL)


def test_unrecorded_spawn_identity_remains_unresolved_and_cannot_receive_terminal_receipt(tmp_path,monkeypatch):
    job,v,root,worker=fixture(tmp_path,'raise SystemExit(0)\n');launch=job/'launch';launch.mkdir()
    selected={'schema':'ovl.pod-job-launch-intent.v1','job_sha256':root,'worker_sha256':worker}
    write_json(launch/'intent.json',selected);write_json(launch/'execution-intent.json',selected)
    assert m.supervision(job,root,worker)['state']=='CHILD_IDENTITY_UNKNOWN'
    signals=[];monkeypatch.setattr(m,'signal_owned',lambda *a:signals.append(a))
    with pytest.raises(m.Refusal,match='identity unknown'):m.abandon(job,root,worker)
    assert not signals and not(job/'abandoned.json').exists()


def test_empty_launch_fence_is_explicitly_abandoned_without_authorizing_retry(tmp_path):
    job,v,root,worker=fixture(tmp_path,'raise SystemExit(0)\n');(job/'launch').mkdir()
    assert m.supervision(job,root,worker)['state']=='LAUNCH_FENCE_WITHOUT_INTENT'
    assert m.abandon(job,root,worker)['observed_child'] is None
    assert start(job,root,worker).returncode!=0


def test_closed_worker_terminal_schemas_match_publisher_canonical_encoding():
    from ovl_pipeline.canonical import canonical
    for code in (-255,-9,0,1,125,255):
        value={'schema':'ovl.workload-job-exit.v1','job_sha256':'a'*64,'state':'EXITED','exit_code':code}
        assert m.encoded(value)==canonical(value)
    for child in (None,{'pid':123,'start_ticks':987654321,'process_group':123}):
        value={'schema':'ovl.workload-job-abandonment.v1','job_sha256':'a'*64,'state':'ABANDONED','observed_child':child,
               'exit_code':'UNAVAILABLE','scope':'supervisor absent; recorded child stopped or absent; no successful computation claim'}
        assert m.encoded(value)==canonical(value)
