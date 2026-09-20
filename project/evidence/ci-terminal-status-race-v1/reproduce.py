"""Force the existing exit-before-final-status window in an actual local worker."""
from pathlib import Path
import sys,threading,time
sys.path[:0]=['src','scripts','tests']
import pod_job_worker as worker
from test_pod_job_worker import fixture,exited
from ovl_pipeline.canonical import read_json,write_json

base=Path('.ovllm-cache/worker-status-race-reproduction-v1').resolve();base.mkdir()
job,value,root,source=fixture(base,'import os,time\nos.write(1,b"x"*(17*1024*1024))\ntime.sleep(20)\n')
(job/'launch').mkdir()
worker.save(job/'launch/intent.json',{'schema':'ovl.pod-job-launch-intent.v1','job_sha256':root,'worker_sha256':source})
pending=threading.Event();release=threading.Event();errors=[];original=worker.save
def paused_save(path,value,**kwargs):
    if Path(path).name=='status.json' and value.get('state')=='EXITED':
        pending.set()
        if not release.wait(5):raise RuntimeError('test synchronization expired')
    return original(path,value,**kwargs)
worker.save=paused_save
def run():
    try:worker.run(job,root,source)
    except BaseException as error:errors.append(type(error).__name__)
thread=threading.Thread(target=run);thread.start()
try:
    assert pending.wait(8),'final status write not observed'
    terminal=exited(job)
    prior=read_json(job/'status.json') if (job/'status.json').exists() else None
    assert terminal['exit_code']<0 and (prior is None or prior['state']!='EXITED')
finally:
    release.set();thread.join(8);worker.save=original
assert not thread.is_alive() and not errors
final=read_json(job/'status.json');assert final['state']=='EXITED' and final['stop_reason']=='storage-bound'
report={'schema':'ovl.worker-status-race-reproduction.v1','result':'PASS','observed_epoch':int(time.time()),
        'worker_sha256':source,'terminal_while_final_status_blocked':terminal,'prior_status':prior,'final_status':final,
        'scope':'Actual local worker and owned child; test instrumentation paused only final status publication after durable exit. Demonstrates possible race, not proof of the unretained CI timing.'}
write_json(base/'verification.json',report);print(report)
