#!/usr/bin/env python3
"""Create a minimal verifier installation separate from the numerical target.

Invoke with the trusted container interpreter using -I -S and pinned public
inputs. Only the selected rfc8785 and packaging wheels enter this installation.
The public interpreter and each installed package are audited before use.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def setup(config,expected,inputs,runtime,verifier,output,*,deadline=None):
    limit=time.monotonic()+max(0,(deadline-time.time()) if deadline is not None else 3600)
    # Import this neighboring, separately pinned public bootstrap script only.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from pod_runtime_setup import selected,sha,isolated_install_command,bounded_install
    runtime,verifier,output=(Path(p).absolute() for p in (runtime,verifier,output))
    for path in (runtime,verifier,output):
        if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('symlink setup path')
    for a,b in ((runtime,verifier),(runtime,output),(verifier,output)):
        if a==b or a in b.parents or b in a.parents:raise ValueError('separate setup trees required')
    if verifier.exists() or output.exists():raise ValueError('fresh verifier and evidence required')
    value,source,wheels,lock,_,archive=selected(config,expected,inputs,wheel_cache=runtime/'wheels')
    from ovl_pipeline.canonical import write_json,digest
    from ovl_pipeline.python_origin import manifest,audit
    from ovl_pipeline.runtime_audit import wheel_manifest,verify_installed
    from packaging.utils import parse_wheel_filename
    python_root=runtime/'public-python'
    origin=manifest(archive,value['interpreter_sha256']);checked=audit(origin,python_root)
    python=python_root/'python/bin/python3.12'
    output.mkdir(parents=True);verifier.mkdir(parents=True)
    selected_wheels=verifier/'wheels';selected_wheels.mkdir()
    rows=[]
    for item in value['bootstrap_wheels']:
        original=wheels/item['path'];destination=selected_wheels/item['path']
        shutil.copyfile(original,destination)
        if sha(destination)!=item['sha256']:raise ValueError('verifier wheel copy changed')
        name,version,_,_=parse_wheel_filename(item['path'])
        rows.append(str(name)+'=='+str(version)+' --hash=sha256:'+item['sha256']+'\n')
    parent_lock=verifier/'requirements.lock';parent_lock.write_text(''.join(sorted(rows)))
    package_manifest=wheel_manifest(parent_lock,selected_wheels)
    venv=verifier/'venv'
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(verifier),'PIP_CONFIG_FILE':'/dev/null',
         'PIP_NO_INDEX':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONDONTWRITEBYTECODE':'1'}
    bounded_install([*isolated_install_command(python,'venv',output,env),'--without-pip',str(venv)],env=env,check=True,deadline=limit)
    bounded_install([*isolated_install_command(python,'pip',output,env),'--python',str(venv/'bin/python'),'install',
                    '--no-index','--no-deps','--no-compile','--require-hashes',
                    '--find-links',str(selected_wheels),'-r',str(parent_lock)],env=env,check=True,deadline=limit)
    installed=verify_installed(package_manifest,{'site':venv/'lib/python3.12/site-packages',
        'prefix':venv,'scripts':venv/'bin','headers':venv/'include/python3.12'})
    write_json(output/'python-payloads.json',origin);write_json(output/'python-audit.json',checked)
    write_json(output/'wheel-payloads.json',package_manifest);write_json(output/'installed-audit.json',installed)
    result={'schema':'ovl.minimal-verifier-setup.v1','result':'PASS','config_sha256':expected,
       'interpreter_archive_sha256':value['interpreter_sha256'],
       'wheel_manifest_sha256':digest(package_manifest),'installed_audit_sha256':digest(installed),
       'bootstrap_executable_sha256':sha(Path('/proc/self/exe')),
       'scope':'separate minimal external verifier; target numerical execution and production acceptance NOT_RUN'}
    write_json(output/'setup.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','inputs','runtime','verifier','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--deadline',type=int)
    args=parser.parse_args()
    if not(sys.flags.isolated and sys.flags.no_site):parser.error('bootstrap requires -I -S')
    setup(args.config,args.config_sha256,args.inputs,args.runtime,args.verifier,args.output,deadline=args.deadline)


if __name__=='__main__':main()
