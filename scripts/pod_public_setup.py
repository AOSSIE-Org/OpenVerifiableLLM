#!/usr/bin/env python3
"""Bounded public downloads followed by the separately audited offline installer.

All executable helpers and source members are selected by the caller's pinned
configuration. The archive supplies only regular source bytes, never paths to trust.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def path(root,name):
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*',name) or any(p in ('.','..') for p in name.split('/')):
        raise ValueError('selected path required')
    result=root/name
    if any(p.is_symlink() for p in [result,*result.parents]):raise ValueError('symlink refused')
    return result


def read(file,expected):
    if not re.fullmatch('[0-9a-f]{64}',expected) or sha(file)!=expected:raise ValueError('selected digest differs')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate key')
            d[k]=v
        return d
    return json.loads(file.read_bytes(),object_pairs_hook=pairs)


def extract(archive,expected,root,files):
    if sha(archive)!=expected:raise ValueError('source archive differs')
    if root.exists() or root.is_symlink():raise ValueError('fresh source required')
    if type(files) is not list or not 1<=len(files)<=10000:raise ValueError('bounded source inventory required')
    names=[];total=0
    for item in files:
        if set(item)!={'path','bytes','sha256'} or type(item['bytes']) is not int or not 0<=item['bytes']<=16*1024**2:raise ValueError('source member bound')
        p=path(root,item['path'])
        if '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo'):raise ValueError('source bytecode refused')
        if not re.fullmatch('[0-9a-f]{64}',item['sha256']):raise ValueError('source digest')
        names.append(item['path']);total+=item['bytes']
    if names!=sorted(set(names)) or total>64*1024**2:raise ValueError('unique sorted bounded source required')
    selected=set(names)
    for parts in (n.split('/') for n in names):
        if any('/'.join(parts[:i]) in selected for i in range(1,len(parts))):raise ValueError('source parent collision')
    root.mkdir(mode=0o700,parents=True)
    # Read in stream order; no extractall, links, metadata, or filesystem names
    # from the tar header become authority over the selected inventory.
    with tarfile.open(archive,'r|gz') as tar:
        for item in files:
            member=tar.next()
            if member is None or not member.isfile() or member.name!=item['path'] or member.size!=item['bytes'] or member.pax_headers:
                raise ValueError('source archive member differs')
            dest=path(root,item['path']);dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            with tar.extractfile(member) as source,dest.open('xb') as target:
                while True:
                    b=source.read(min(1024**2,item['bytes']-count+1))
                    if not b:break
                    count+=len(b)
                    if count>item['bytes']:raise ValueError('source length')
                    h.update(b);target.write(b)
                target.flush();os.fsync(target.fileno())
            if count!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('source bytes differ')
        if tar.next() is not None:raise ValueError('unlisted source member')


def setup(config,expected,inputs,runtime,output,deadline,*,execute=subprocess.run):
    started=time.time();monotonic_end=time.monotonic()+deadline-started
    inputs=Path(inputs).absolute();runtime=Path(runtime).absolute();output=Path(output).absolute()
    for p in (inputs,runtime,output,Path(config).absolute()):
        if any(a.is_symlink() for a in [p,*p.parents]):raise ValueError('setup symlink')
    if type(deadline) is not int or not 0<deadline-time.time()<=300:raise ValueError('bounded setup deadline required')
    if runtime.exists() or output.exists():raise ValueError('fresh runtime and evidence required')
    if any(a==b or a in b.parents or b in a.parents for a,b in ((inputs,runtime),(inputs,output),(runtime,output))):raise ValueError('separate setup roots required')
    value=read(Path(config),expected)
    keys={'schema','offline_config','offline_config_sha256','fetch_script','fetch_script_sha256','setup_script','setup_script_sha256','wheel_plan','wheel_plan_sha256','source_archive','source_archive_sha256','download_seconds'}
    if type(value) is not dict or set(value)!=keys or value['schema']!='ovl.public-runtime-setup.v1':raise ValueError('public setup schema')
    for key in ('offline_config','fetch_script','setup_script','wheel_plan','source_archive'):
        if sha(path(inputs,value[key]))!=value[key+'_sha256']:raise ValueError('selected setup input differs')
    offline=read(path(inputs,value['offline_config']),value['offline_config_sha256'])
    if type(offline) is not dict or offline.get('schema')!='ovl.offline-runtime-setup.v1':raise ValueError('offline schema')
    source=path(inputs,offline['source_root']);wheels=path(inputs,offline['wheels'])
    if source==wheels or source in wheels.parents or wheels in source.parents or wheels.exists():raise ValueError('fresh distinct source and wheels required')
    if type(value['download_seconds']) is not int or not 1<=value['download_seconds']<=180:raise ValueError('download time bound')
    output.mkdir(mode=0o700,parents=True)
    with (output/'selected-config.json').open('xb') as f:
        f.write(Path(config).read_bytes());f.flush();os.fsync(f.fileno())
    extract(path(inputs,value['source_archive']),value['source_archive_sha256'],source,offline['source_files'])
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(runtime),'PYTHONDONTWRITEBYTECODE':'1'}
    def run(args):
        left=min(deadline-time.time(),monotonic_end-time.monotonic())
        if left<=0:raise TimeoutError('original setup deadline expired')
        execute([sys.executable,'-I','-S',*args],env=env,check=True,timeout=left)
    download_deadline=min(deadline,int(time.time())+value['download_seconds'])
    run([str(path(inputs,value['fetch_script'])),'--plan',str(path(inputs,value['wheel_plan'])),'--plan-sha256',value['wheel_plan_sha256'],
         '--output',str(wheels),'--report',str(output/'downloads.json'),'--deadline',str(download_deadline)])
    run([str(path(inputs,value['setup_script'])),'setup','--config',str(path(inputs,value['offline_config'])),'--config-sha256',value['offline_config_sha256'],
         '--inputs',str(inputs),'--runtime',str(runtime),'--output',str(output/'offline')])
    if min(deadline-time.time(),monotonic_end-time.monotonic())<=0:raise TimeoutError('original setup deadline expired')
    downloads=json.loads((output/'downloads.json').read_bytes())
    installed=json.loads((output/'offline/setup.json').read_bytes())
    plan=read(path(inputs,value['wheel_plan']),value['wheel_plan_sha256'])
    if (not plan.get('files') or downloads.get('schema')!='ovl.public-wheel-download-result.v1' or downloads.get('plan_sha256')!=value['wheel_plan_sha256']
        or type(downloads.get('files')) is not list or len(downloads['files'])!=len(plan['files'])):raise ValueError('download receipt parent differs')
    for expected_file,actual in zip(plan['files'],downloads['files']):
        if any(actual.get(k)!=v for k,v in expected_file.items()) or actual.get('result')!='COMPLETE_HASH_MATCH':raise ValueError('incomplete selected download receipt')
    if (installed.get('schema')!='ovl.offline-runtime-setup-result.v1' or installed.get('result')!='PASS'
        or installed.get('config_sha256')!=value['offline_config_sha256']):raise ValueError('offline audit receipt differs')
    result={'schema':'ovl.public-runtime-setup-result.v1','result':'PASS','config_sha256':expected,
            'downloads_sha256':sha(output/'downloads.json'),'offline_result_sha256':sha(output/'offline/setup.json'),
            'operator_observed_start_epoch':int(started),'operator_observed_finish_epoch':int(time.time()),'original_deadline_epoch':deadline,
            'scope':'selected public downloads and audited offline runtime only; no CUDA/training acceptance'}
    with (output/'setup.json').open('x') as f:json.dump(result,f,sort_keys=True,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','inputs','runtime','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--config-sha256',required=True);p.add_argument('--deadline',required=True,type=int)
    a=p.parse_args()
    if not(sys.flags.isolated and sys.flags.no_site):p.exit(1,'bootstrap requires -I -S\n')
    # Keep the caller's original outer deadline and additionally bound this
    # one-shot stage to 300s from entry; this never extends the rental/job bound.
    setup(a.config,a.config_sha256,a.inputs,a.runtime,a.output,min(a.deadline,int(time.time())+300))


if __name__=='__main__':main()
