"""Actual detached worker, caller death and recovery; synthetic local scope."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import pytest
from ovl_pipeline.canonical import EvidenceError, digest, file_hash, read_json, write_json
from ovl_pipeline.lifecycle import Pending
from ovl_pipeline.lifecycle_process import observe,submit,recover,live
from ovl_pipeline.training import code_root


def request(tmp_path):
    source=Path(__file__).resolve().parents[1]
    raw=source/'tests/fixtures/pipeline'
    output=tmp_path/'run'
    return {'schema':'ovl.lifecycle-process.v1','module':'ovl_pipeline.lifecycle_fixture',
            'arguments':['run','--source',str(raw),'--run',str(output)],
            'source_root':str(source),'source_sha256':code_root(),'deadline':int(time.time())+45,
            'output':str(output),'inputs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)}
                                        for p in (raw/'wiki.xml',raw/'conversations.json')]}


def finish(directory,r):
    end=time.time()+40
    while time.time()<end:
        result=observe(directory,r)
        if result['status'] in ('complete','failed'):
            assert result['status']=='complete',(directory/'stderr.log').read_text()
            return result
        time.sleep(.05)
    raise AssertionError('worker did not finish')


def cleanup(directory):
    for name in ('child.json','process.json'):
        path=directory/name
        if path.exists():
            identity=read_json(path)
            if live(identity): os.kill(identity['pid'],signal.SIGKILL)


def test_caller_dies_after_submission_worker_completes_once(tmp_path,monkeypatch):
    r=request(tmp_path);job=tmp_path/'job';selection=tmp_path/'request.json';write_json(selection,r)
    helper=tmp_path/'caller.py'
    helper.write_text('''import os,sys
from pathlib import Path
from ovl_pipeline.canonical import read_json
from ovl_pipeline.lifecycle_process import submit
if __name__=='__main__':
 def event(name):
  if name=='submitted':os._exit(73)
 submit(Path(sys.argv[1]),read_json(Path(sys.argv[2])),event=event)
''')
    env={**os.environ,'PYTHONPATH':str(Path(r['source_root'])/'src')}
    if os.environ.get('OVL_TEST_NUMERIC_RUNNER'):
        env['OVL_WORKLOAD_NUMERIC_RUNNER']=os.environ['OVL_TEST_NUMERIC_RUNNER']
    try:
        caller=subprocess.run([sys.executable,str(helper),str(job),str(selection)],env=env,timeout=15)
        assert caller.returncode==73
        with pytest.raises(Pending): submit(job,r)
        result=finish(job,r)
        assert result['exit_code']==0 and not result['deadline_expired']
        closure=read_json(Path(r['output'])/'objects/close/closure.json')
        assert closure['result']=='PASS' and closure['production_acceptance']=='NOT_RUN'
        assert observe(job,r)==result
    finally:cleanup(job)


def test_recover_after_supervisor_and_child_death_preserves_partial(tmp_path,monkeypatch):
    r=request(tmp_path);job=tmp_path/'job'
    if os.environ.get('OVL_TEST_NUMERIC_RUNNER'):
        monkeypatch.setenv('OVL_WORKLOAD_NUMERIC_RUNNER',os.environ['OVL_TEST_NUMERIC_RUNNER'])
    try:
        submit(job,r)
        end=time.time()+15
        while not (Path(r['output'])/'private/journal.json').exists() and time.time()<end:
            time.sleep(.005)
        assert (Path(r['output'])/'private/journal.json').exists()
        assert not (job/'terminal.json').exists()
        cleanup(job)
        limit=time.time()+5
        while time.time()<limit:
            try:
                started=recover(job,r)
                break
            except Pending:time.sleep(.05)
        else:raise AssertionError('dead workload still owns lease')
        assert started['status']=='submitted'
        result=finish(job,r)
        assert result['exit_code']==0
        assert list((job/'recovery').glob('*/child.json'))
        assert read_json(Path(r['output'])/'objects/close/closure.json')['result']=='PASS'
    finally:cleanup(job)


def test_workload_identity_and_original_deadline_are_strict(tmp_path):
    r=request(tmp_path);r['deadline']=int(time.time())-1
    with pytest.raises(EvidenceError,match='deadline'):submit(tmp_path/'job',r)
    r=request(tmp_path);r['source_sha256']='f'*64
    with pytest.raises(EvidenceError,match='source'):submit(tmp_path/'job',r)


def test_supervisor_loss_kills_numerical_child_and_permits_recovery(tmp_path,monkeypatch):
    r=request(tmp_path);job=tmp_path/'job'
    if os.environ.get('OVL_TEST_NUMERIC_RUNNER'):
        monkeypatch.setenv('OVL_WORKLOAD_NUMERIC_RUNNER',os.environ['OVL_TEST_NUMERIC_RUNNER'])
    try:
        submit(job,r)
        end=time.time()+10
        while not (job/'child.json').exists() and time.time()<end:time.sleep(.005)
        assert (job/'child.json').exists()
        supervisor=read_json(job/'process.json');os.kill(supervisor['pid'],signal.SIGKILL)
        end=time.time()+5
        while live(read_json(job/'child.json')) and time.time()<end:time.sleep(.02)
        assert not live(read_json(job/'child.json'))
        assert observe(job,r)['status']=='uncertain'
        assert recover(job,r)['status']=='submitted'
        assert finish(job,r)['exit_code']==0
    finally:cleanup(job)


def test_actual_child_is_stopped_at_original_deadline(tmp_path,monkeypatch):
    r=request(tmp_path);r['deadline']=int(time.time())+2;job=tmp_path/'job'
    if os.environ.get('OVL_TEST_NUMERIC_RUNNER'):
        monkeypatch.setenv('OVL_WORKLOAD_NUMERIC_RUNNER',os.environ['OVL_TEST_NUMERIC_RUNNER'])
    try:
        submit(job,r)
        end=time.time()+10
        while not (job/'terminal.json').exists() and time.time()<end:time.sleep(.05)
        result=observe(job,r)
        assert result['status']=='failed' and result['deadline_expired']
        if (job/'child.json').exists():assert not live(read_json(job/'child.json'))
        with pytest.raises(EvidenceError,match='deadline'):recover(job,r)
    finally:cleanup(job)


def test_long_uptime_and_interrupted_clock_initialization_preserve_bound(tmp_path,monkeypatch):
    import ovl_pipeline.lifecycle_process as m
    r=request(tmp_path);job=tmp_path/'job'
    monkeypatch.setattr(m.time,'monotonic',lambda:2**53/1e9+1000)
    monkeypatch.setattr(m,'spawn_supervisor',lambda *a:{'status':'submitted'})
    original=m.write_json
    def interrupted(path,value):
        if Path(path).name=='request.json':raise InterruptedError('synthetic pre-request loss')
        original(path,value)
    monkeypatch.setattr(m,'write_json',interrupted)
    with pytest.raises(InterruptedError):m.submit(job,r)
    bound=read_json(job/'clock.json')
    assert 0<bound['stop_monotonic_ms']<2**53
    monkeypatch.setattr(m,'write_json',original)
    assert m.submit(job,r)['status']=='submitted'
    assert read_json(job/'clock.json')==bound


def audited_request(tmp_path):
    r=request(tmp_path);root=Path(r['output']);source=Path(r['source_root'])
    outer={'--lock':str(tmp_path/'gpu.lock'),'--wheels':str(tmp_path/'wheels'),
           '--venv':str(tmp_path/'venv'),'--source':str(source/'src'),'--output':str(root/'audit'),
           '--module':'ovl_pipeline.production_record','--interpreter-archive':str(tmp_path/'python.tar.gz'),
           '--interpreter-sha256':'a'*64,'--interpreter-root':str(tmp_path/'python')}
    r['module']='ovl_pipeline.runtime_launch'
    r['arguments']=[x for pair in outer.items() for x in pair]+['--','--output',str(root/'result')]
    return r


def test_completed_record_reconciles_actual_audit_receipt_without_resuming(tmp_path,monkeypatch):
    import ovl_pipeline.lifecycle_process as m
    r=audited_request(tmp_path);job=tmp_path/'job';calls=[]
    monkeypatch.setattr(m,'spawn_supervisor',lambda *args:calls.append(args) or {'status':'submitted'})
    submit(job,r)
    output=Path(r['output']);audit=output/'audit'
    launch={'module':'ovl_pipeline.production_record','arguments':r['arguments'][r['arguments'].index('--')+1:],
            'source':str(Path(r['source_root'])/'src')}
    write_json(audit/'launch.json',launch)
    write_json(audit/'process.json',{'schema':'ovl.audited-runtime-process.v1','launch_sha256':digest(launch),'exit_code':0})
    write_json(output/'result/record.json',{'scope':'synthetic process protocol fixture; no scientific credit'})
    monkeypatch.setattr(m.time,'time',lambda:r['deadline']+10)
    result=recover(job,r)
    assert result['status']=='complete' and result['exit_code']==0 and result['deadline_expired'] is None
    assert len(calls)==1 and 'supervisor exit unknown' in result['scope']
    assert observe(job,r)==result


def test_completed_record_without_execution_receipt_remains_uncertain(tmp_path,monkeypatch):
    import ovl_pipeline.lifecycle_process as m
    r=audited_request(tmp_path);job=tmp_path/'job'
    monkeypatch.setattr(m,'spawn_supervisor',lambda *args:{'status':'submitted'})
    submit(job,r);write_json(Path(r['output'])/'result/record.json',{'scope':'synthetic'})
    with pytest.raises(Pending,match='never resume completed'):recover(job,r)
    assert not (job/'terminal.json').exists()
