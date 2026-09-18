#!/usr/bin/env python3
"""Fresh audited CUDA fixture record, complete replay and separate resume probe.

This bounded synthetic probe is not a throughput measurement or production run.
The pinned offline launcher rehashes source, wheels, installation and interpreter
before each independent numerical process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import uuid


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def probe(setup_script,setup_sha256,config,config_sha256,inputs,runtime,stream,recipe,kernel,output,deadline,*,execute=subprocess.run):
    for p in (setup_script,config,inputs,runtime,stream,recipe,kernel,output):
        if not p.is_absolute() or any(v.is_symlink() for v in [p,*p.parents]):raise ValueError('absolute regular selected paths required')
    if sha(setup_script)!=setup_sha256 or sha(config)!=config_sha256:raise ValueError('selected launcher/config differs')
    if output.exists():raise ValueError('fresh probe output required')
    if os.environ.get('OVL_ACTIVITY_FILE')!=str(output/'activity.json'):
        raise ValueError('selected completed-phase activity path required')
    if not 0<deadline-time.time()<=1500:raise ValueError('bounded original probe deadline required')
    # Every independently audited process is material work. Only successfully
    # checked phase reports advance the bounded prefix below; never manufacture
    # numerical liveness by combining different child process identities.
    deadline=min(deadline,int(time.time())+720)
    monotonic_end=time.monotonic()+deadline-time.time()
    output.mkdir(mode=0o700,parents=True)
    process_instance=uuid.uuid4().hex;completed=[]
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(runtime),'PYTHONDONTWRITEBYTECODE':'1'}
    def observed(phase,report):
        if min(deadline-time.time(),monotonic_end-time.monotonic())<=0:raise TimeoutError('original probe deadline expired')
        completed.append({'phase':phase,'report_sha256':sha(report)})
        value={'schema':'ovl.audited-pilot-phases.v1','process_instance':process_instance,'pid':os.getpid(),
               'completed':completed,'scope':'operator-supervision-only-not-training-verification'}
        path=output/'activity.json'
        if path.is_symlink():raise ValueError('activity symlink')
        fd,name=tempfile.mkstemp(prefix='.activity-',dir=output)
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def run(name,args):
        remaining=min(deadline-time.time(),monotonic_end-time.monotonic())
        if remaining<=0:raise TimeoutError('original probe deadline expired')
        command=[sys.executable,'-I','-S',str(setup_script),'launch','--config',str(config),'--config-sha256',config_sha256,
                 '--inputs',str(inputs),'--runtime',str(runtime),'--output',str(output/(name+'-launch')),'--module','ovl_pipeline.gpu_pilot','--',*args]
        # v4's first fully audited phase took ~135s. Allow 240s per phase,
        # below the unchanged 300s useful-progress guard; the combined 720s
        # ceiling and original selected job deadline still dominate.
        execute(command,env=env,check=True,timeout=min(remaining,240))
    run('record',['record','--recipe',str(recipe),'--kernel',str(kernel),'--updates','8','--warmup-updates','2','--checkpoint-every','4',
                  '--stream',str(stream),'--output',str(output/'record')])
    record=output/'record/record.json';record_sha=sha(record)
    r=json.loads(record.read_bytes())
    if (r.get('schema')!='ovl.gpu-pilot-record.v1' or r.get('result')!='RECORDED_NOT_REPLAYED'
        or r['updates']!=8 or r['eligible_duration_for_forecast'] is not False or len(r['boundaries'])!=3):raise ValueError('unexpected tiny record')
    observed('record',record)
    args=['replay','--record-directory',str(output/'record'),'--expected-record-sha256',record_sha,'--stream',str(stream)]
    reports={}
    for name,count,scope in (('replay',8,'fresh-initialization-continuous-pilot-replay'),('resume',4,'training-resume-continuation-probe')):
        run(name,[*args,'--output',str(output/name),*(['--resume-from','1'] if name=='resume' else [])])
        report=output/name/'verification.json';value=json.loads(report.read_bytes());reports[name]=value
        if (value.get('schema')!='ovl.gpu-pilot-replay.v1' or value['result']!='PASS' or value['scope']!=scope or value['updates_recomputed']!=count or value['record_sha256']!=record_sha
            or value['initial_state_regenerated'] is not True or value['independent_third_party'] is not False):raise ValueError('incomplete selected probe')
        if sha(record)!=record_sha:raise ValueError('record changed')
        if name=='replay' and len(value['compared'])!=3:raise ValueError('incomplete pilot boundary comparison')
        if value.get('resume_from')!=(1 if name=='resume' else None):raise ValueError('wrong selected resume boundary')
        # This particular resume regenerates zero, restores boundary one, then
        # recomputes to boundary two: all three comparisons must equal replay.
        if name=='resume' and value['compared']!=reports['replay']['compared']:raise ValueError('probe compared states differ')
        observed(name,report)
    v=reports['replay'];s=reports['resume']
    if s['compared'][-1]!=v['compared'][-1] or len(v['compared'])!=3:raise ValueError('probe final state differs')
    if sha(record)!=record_sha or min(deadline-time.time(),monotonic_end-time.monotonic())<=0:raise ValueError('record changed or original deadline expired')
    result={'schema':'ovl.tiny-cuda-probe.v1','result':'PASS','record_sha256':record_sha,
            'replay_sha256':sha(output/'replay/verification.json'),'resume_sha256':sha(output/'resume/verification.json'),
            'scope':'synthetic eight-update actual CUDA record/full fresh-process replay and separate resume only',
            'production_admission':'NOT_RUN','throughput_forecast':'NOT_RUN','independent_third_party':False}
    with (output/'probe.json').open('x') as f:json.dump(result,f,sort_keys=True,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('setup-script','config','inputs','runtime','stream','recipe','kernel','output'):p.add_argument('--'+name,required=True,type=Path)
    for name in ('setup-sha256','config-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline',required=True,type=int);a=p.parse_args()
    if not(sys.flags.isolated and sys.flags.no_site):p.exit(1,'probe launcher requires -I -S\n')
    probe(**vars(a))


if __name__=='__main__':main()
