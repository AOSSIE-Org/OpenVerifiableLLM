#!/usr/bin/env python3
"""Offline public runtime setup and audited launch from operator-selected inputs.

Invoke the digest-pinned script with the container's trusted bootstrap interpreter
using -I -S. That interpreter is trusted transitively through the selected image;
this is not proof of a reproducible container/CPython build or remote attestation.
No credentials, package indexes or paid external services are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def confined(root,name):
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*',name) or any(p in ('.','..') for p in name.split('/')):
        raise ValueError('invalid input path')
    p=root/name
    for parent in [p,*p.parents]:
        if parent.is_symlink():raise ValueError('symlink input')
    return p


def selected(config_file,expected,inputs):
    inputs=Path(inputs).absolute();config_file=Path(config_file)
    if not re.fullmatch('[0-9a-f]{64}',expected) or config_file.is_symlink() or sha(config_file)!=expected:
        raise ValueError('setup configuration differs from operator selection')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate configuration key')
            d[k]=v
        return d
    value=json.loads(config_file.read_bytes(),object_pairs_hook=pairs)
    keys={'schema','source_root','source_files','dependency_lock','interpreter_archive','interpreter_sha256','wheels','bootstrap_wheels'}
    if type(value) is not dict or set(value)!=keys or value['schema']!='ovl.offline-runtime-setup.v1':raise ValueError('setup configuration schema')
    source=confined(inputs,value['source_root']);names=[]
    for item in value['source_files']:
        if set(item)!={'path','bytes','sha256'} or type(item['bytes']) is not int:raise ValueError('source inventory shape')
        path=confined(source,item['path'])
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('source bytes differ')
        names.append(item['path'])
    actual=[]
    for path in source.rglob('*'):
        if path.is_symlink():raise ValueError('source symlink')
        if '__pycache__' in path.relative_to(source).parts or path.suffix in ('.pyc','.pyo'):
            raise ValueError('bytecode in selected source')
        if path.is_file():actual.append(path.relative_to(source).as_posix())
    if names!=sorted(set(names)) or sorted(actual)!=names:raise ValueError('source inventory is not complete')
    wheels=confined(inputs,value['wheels']);imports=value['bootstrap_wheels']
    if type(imports) is not list or len(imports)!=2:raise ValueError('explicit bootstrap wheel pair required')
    packages=[];paths=[]
    for item in imports:
        if set(item)!={'path','sha256'}:raise ValueError('bootstrap wheel shape')
        path=confined(wheels,item['path']);package=path.name.split('-')[0]
        if package not in ('rfc8785','packaging') or path.suffix!='.whl' or sha(path)!=item['sha256']:raise ValueError('bootstrap wheel differs')
        packages.append(package);paths.append(str(path))
    if sorted(packages)!=['packaging','rfc8785']:raise ValueError('bootstrap wheel selection differs')
    lock=confined(source,value['dependency_lock'])
    # Break the bootstrap parser dependency before importing either helper:
    # only exact pure-Python helper rows and their own lock hashes authorize it.
    rows=[];pending=''
    for raw in lock.read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#') or line.startswith('--index-url ') or line.startswith('--extra-index-url '):continue
        pending+=line[:-1]+' ' if line.endswith('\\') else line
        if line.endswith('\\'):continue
        rows.append(pending);pending=''
    if pending:raise ValueError('unfinished bootstrap lock')
    for item,package in zip(imports,packages):
        selected_rows=[row for row in rows if re.match(re.escape(package)+r'==[^ ]+ ',row)]
        if (len(selected_rows)!=1 or item['sha256'] not in re.findall(r'--hash=sha256:([0-9a-f]{64})(?= |$)',selected_rows[0])
            or not re.search(r'-(?:py3|py2\.py3)-none-any\.whl$',item['path'])):
            raise ValueError('bootstrap wheel is not authorized by its own pure-Python lock row')
    # These two selected pure-Python public wheels are the trusted parent imports.
    # Target numerical packages are not imported before the full external audit.
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(source/'src'),*paths]
    from ovl_pipeline.runtime_audit import wheel_manifest
    manifest=wheel_manifest(lock,wheels)
    archive=confined(inputs,value['interpreter_archive'])
    if sha(archive)!=value['interpreter_sha256']:raise ValueError('public interpreter archive differs')
    return value,source,wheels,lock,manifest,archive


def layout(runtime,output):
    runtime=Path(runtime).absolute();output=Path(output).absolute()
    for path in (runtime,output):
        if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('runtime/output symlink')
    if runtime==output or runtime in output.parents or output in runtime.parents:raise ValueError('runtime and evidence must be separate')
    if output.exists():raise ValueError('setup/launch evidence requires fresh output')
    return runtime,output


def setup(config_file,expected,inputs,runtime,output,*,execute=subprocess.run):
    runtime,output=layout(runtime,output)
    if runtime.exists():raise ValueError('runtime setup requires a fresh tree; preserve partial attempts')
    value,source,wheels,lock,manifest,archive=selected(config_file,expected,inputs)
    from ovl_pipeline.canonical import digest,write_json
    from ovl_pipeline.python_origin import extract
    from ovl_pipeline.runtime_audit import verify_installed
    from ovl_pipeline.runtime_launch import launch
    output.mkdir(mode=0o700,parents=True);runtime.mkdir(mode=0o700,parents=True)
    write_json(output/'selected-config.json',value);write_json(output/'wheel-payloads.json',manifest)
    python_root=runtime/'public-python';payloads,checked=extract(archive,value['interpreter_sha256'],python_root)
    write_json(output/'python-payloads.json',payloads);write_json(output/'python-audit.json',checked)
    python=python_root/'python/bin/python3.12';venv=runtime/'venv'
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(runtime),'PIP_CONFIG_FILE':'/dev/null',
         'PIP_NO_INDEX':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONDONTWRITEBYTECODE':'1'}
    execute([str(python),'-I','-m','venv','--without-pip',str(venv)],env=env,check=True)
    install=output/'offline-install.txt'
    # Use the exact selected local wheels, including Torch's direct-URL lock row.
    # A direct URL in pip's original requirements would bypass --no-index.
    install.write_text(''.join(str(p.resolve())+' --hash=sha256:'+sha(p)+'\n' for p in sorted(wheels.glob('*.whl'))))
    execute([str(python),'-I','-m','pip','--python',str(venv/'bin/python'),'install','--no-index','--no-deps','--no-compile',
             '--require-hashes','-r',str(install)],env=env,check=True)
    installed=verify_installed(manifest,{'site':venv/'lib/python3.12/site-packages','prefix':venv,'scripts':venv/'bin','headers':venv/'include/python3.12'})
    write_json(output/'installed-audit.json',installed)
    inspected=launch(lock,wheels,venv,source/'src',output/'audited-inspection','ovl_pipeline.runtime_launch',['--inspect-current'],
                     interpreter_archive=archive,interpreter_sha256=value['interpreter_sha256'],interpreter_root=python_root)
    result={'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':expected,
            'bootstrap_executable_sha256':sha(Path('/proc/self/exe')),'wheel_manifest_sha256':digest(manifest),
            'python_manifest_sha256':digest(payloads),'installed_audit_sha256':digest(installed),'inspection':inspected,
            'scope':'complete public binary/package identity and constrained CPU startup; CUDA admission NOT_RUN',
            'target_executable_sha256':sha(python),'network_installation':'DISABLED','production_acceptance':'NOT_RUN'}
    write_json(output/'setup.json',result);return result


def audited(config_file,expected,inputs,runtime,output,module,arguments):
    runtime,output=layout(runtime,output)
    value,source,wheels,lock,manifest,archive=selected(config_file,expected,inputs)
    from ovl_pipeline.runtime_launch import launch
    return launch(lock,wheels,runtime/'venv',source/'src',output,module,arguments,
                  interpreter_archive=archive,interpreter_sha256=value['interpreter_sha256'],interpreter_root=runtime/'public-python')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['setup','launch'])
    for name in ('config','inputs','runtime','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--config-sha256',required=True);p.add_argument('--module')
    raw=sys.argv[1:];split=raw.index('--') if '--' in raw else len(raw)
    a=p.parse_args(raw[:split]);args=raw[split+1:]
    if not(sys.flags.isolated and sys.flags.no_site):p.exit(1,'bootstrap requires -I -S\n')
    try:
        if a.action=='setup':
            if a.module or args:raise ValueError('setup takes no target command')
            result=setup(a.config,a.config_sha256,a.inputs,a.runtime,a.output)
        else:
            if not a.module:raise ValueError('launch requires selected module')
            result=audited(a.config,a.config_sha256,a.inputs,a.runtime,a.output,a.module,args)
        print(result['schema'])
    except Exception as error:p.exit(1,'runtime setup refused: '+type(error).__name__+'; preserve partial evidence\n')


if __name__=='__main__':main()
