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
    if not 0<deadline-time.time()<=900:raise ValueError('bounded original probe deadline required')
    output.mkdir(mode=0o700,parents=True)
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(runtime),'PYTHONDONTWRITEBYTECODE':'1'}
    def run(name,args):
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('original probe deadline expired')
        command=[sys.executable,'-I','-S',str(setup_script),'launch','--config',str(config),'--config-sha256',config_sha256,
                 '--inputs',str(inputs),'--runtime',str(runtime),'--output',str(output/(name+'-launch')),'--module','ovl_pipeline.gpu_pilot','--',*args]
        execute(command,env=env,check=True,timeout=remaining)
    run('record',['record','--recipe',str(recipe),'--kernel',str(kernel),'--updates','8','--warmup-updates','2','--checkpoint-every','4',
                  '--stream',str(stream),'--output',str(output/'record')])
    record=output/'record/record.json';record_sha=sha(record)
    args=['replay','--record-directory',str(output/'record'),'--expected-record-sha256',record_sha,'--stream',str(stream)]
    run('replay',[*args,'--output',str(output/'replay')])
    run('resume',[*args,'--output',str(output/'resume'),'--resume-from','1'])
    # The numerical module owns these checks; verify its required report scope
    # before this operational wrapper declares completion as well.
    reports={name:json.loads((output/name/file).read_bytes()) for name,file in [('record','record.json'),('replay','verification.json'),('resume','verification.json')]}
    r=reports['record'];v=reports['replay'];s=reports['resume']
    if r['updates']!=8 or r['eligible_duration_for_forecast'] is not False or len(r['boundaries'])!=3:raise ValueError('unexpected tiny record')
    for value,count,scope in ((v,8,'fresh-initialization-continuous-pilot-replay'),(s,4,'training-resume-continuation-probe')):
        if (value['result']!='PASS' or value['scope']!=scope or value['updates_recomputed']!=count or value['record_sha256']!=record_sha
            or value['initial_state_regenerated'] is not True or value['independent_third_party'] is not False):raise ValueError('incomplete selected probe')
    if s['compared'][-1]!=v['compared'][-1] or len(v['compared'])!=3:raise ValueError('probe final state differs')
    if sha(record)!=record_sha or time.time()>=deadline:raise ValueError('record changed or original deadline expired')
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
