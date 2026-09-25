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
import stat
import subprocess
import sys
import time
import tempfile
import uuid


SETUP_PHASES=('source','selection','interpreter','installation','installed',
              'launch-interpreter','launch-wheels','launch-installed','inspection')


class SetupProgress:
    """Completed work only; no timer thread, checkpoint or verification credit."""
    def __init__(self,*,clock=time.monotonic):
        self.clock=clock;self.last=None;self.started=clock();self.completed=[];self.copied=0
        self.instance=uuid.uuid4().hex

    def __call__(self,phase=None,*,copied_bytes=None):
        if phase is not None:
            if phase!=SETUP_PHASES[len(self.completed)]:raise ValueError('setup phase order')
            self.completed.append(phase)
            print('setup completed '+phase+' elapsed_seconds='+str(round(self.clock()-self.started,3)),flush=True)
        if copied_bytes is not None:
            if type(copied_bytes) is not int or not self.copied<=copied_bytes<=4*1024**3:
                raise ValueError('setup copy counter')
            self.copied=copied_bytes
        selected=os.environ.get('OVL_ACTIVITY_FILE')
        if not selected:return
        now=self.clock()
        if phase is None and self.last is not None and 0<=now-self.last<30:return
        path=Path(selected)
        if not path.is_absolute() or path.name!='activity.json' or not path.parent.is_dir() or any(p.is_symlink() for p in [path,*path.parents]):
            raise ValueError('selected setup activity path required')
        value={'schema':'ovl.runtime-setup-activity.v1','process_instance':self.instance,'pid':os.getpid(),
               'completed':list(self.completed),'copied_bytes':self.copied,
               'scope':'operator-supervision-only-not-input-verification'}
        pending=path.with_name('.setup-activity-'+uuid.uuid4().hex)
        try:
            with pending.open('x') as f:
                json.dump(value,f,sort_keys=True,separators=(',',':'));f.flush();os.fsync(f.fileno())
            os.replace(pending,path)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if pending.exists():pending.unlink()
        self.last=now


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


def cache_wheels(source,cache,*,progress=None):
    """Copy bounded regular archives once; all complete wheel audits still run."""
    source=Path(source);cache=Path(cache)
    if any(p.is_symlink() for p in [cache,*cache.parents]):raise ValueError('wheel cache symlink')
    if cache.exists() or source==cache or source in cache.parents or cache in source.parents:
        raise ValueError('fresh separate wheel cache required')
    files=sorted(source.iterdir());total=0;snapshots={}
    def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    if not 1<=len(files)<=1024:raise ValueError('wheel cache file bound')
    for p in files:
        if p.is_symlink() or not p.is_file() or p.suffix!='.whl':raise ValueError('regular wheel archives required')
        snapshots[p.name]=identity(p.stat());total+=snapshots[p.name][2]
    if total>4*1024**3:raise ValueError('wheel cache byte bound')
    cache.mkdir(mode=0o700);copied=0
    for p in files:
        # Never follow a replaced input symlink or accept a growing archive.
        with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as src:
            before=os.fstat(src.fileno());count=0
            if not stat.S_ISREG(before.st_mode) or identity(before)!=snapshots[p.name]:
                raise ValueError('wheel changed before cache copy')
            with (cache/p.name).open('xb') as dst:
                while True:
                    data=src.read(min(1024**2,before.st_size-count+1))
                    if not data:break
                    count+=len(data)
                    if count>before.st_size:raise ValueError('wheel changed during cache copy')
                    dst.write(data);copied+=len(data)
                    if progress is not None:progress(copied_bytes=copied)
                dst.flush();os.fsync(dst.fileno())
            after=os.fstat(src.fileno())
            if count!=before.st_size or identity(before)!=identity(after) or identity(p.lstat())!=identity(before):
                raise ValueError('wheel changed during cache copy')
    if sorted(source.iterdir())!=files:raise ValueError('wheel inventory changed during cache copy')
    # No cache-success or scientific credit: selected() next rebuilds the
    # complete manifest from these copied bytes against the original lock.


def selected(config_file,expected,inputs,*,wheel_cache=None,populate_cache=False,progress=None):
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
    if progress is not None:progress('source')
    wheels=confined(inputs,value['wheels']);imports=value['bootstrap_wheels']
    if type(imports) is not list or len(imports)!=2:raise ValueError('explicit bootstrap wheel pair required')
    if wheel_cache is not None:
        wheel_cache=Path(wheel_cache)
        if any(p.is_symlink() for p in [wheel_cache,*wheel_cache.parents]):raise ValueError('wheel cache symlink')
        if populate_cache:cache_wheels(wheels,wheel_cache,progress=progress)
        if not wheel_cache.is_dir():raise ValueError('missing selected wheel cache')
        wheels=wheel_cache
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
    if progress is not None:progress('selection')
    return value,source,wheels,lock,manifest,archive


def layout(runtime,output):
    runtime=Path(runtime).absolute();output=Path(output).absolute()
    for path in (runtime,output):
        if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('runtime/output symlink')
    if runtime==output or runtime in output.parents or output in runtime.parents:raise ValueError('runtime and evidence must be separate')
    if output.exists():raise ValueError('setup/launch evidence requires fresh output')
    return runtime,output


def isolated_install_command(python,module,cache_parent,environment):
    """Exclude old caches in the parent and pip's delegated interpreter."""
    cache=tempfile.mkdtemp(prefix='setup-cache-',dir=cache_parent)
    environment['PYTHONPYCACHEPREFIX']=cache
    return [str(python),'-I','-B','-X','pycache_prefix='+cache,'-m',module]


def setup(config_file,expected,inputs,runtime,output,*,execute=subprocess.run):
    runtime,output=layout(runtime,output)
    if runtime.exists():raise ValueError('runtime setup requires a fresh tree; preserve partial attempts')
    runtime.mkdir(mode=0o700,parents=True);progress=SetupProgress()
    value,source,wheels,lock,manifest,archive=selected(config_file,expected,inputs,
                                                    wheel_cache=runtime/'wheels',populate_cache=True,progress=progress)
    from ovl_pipeline.canonical import digest,write_json
    from ovl_pipeline.python_origin import extract
    from ovl_pipeline.runtime_audit import verify_installed
    from ovl_pipeline.runtime_launch import launch
    output.mkdir(mode=0o700,parents=True)
    write_json(output/'selected-config.json',value);write_json(output/'wheel-payloads.json',manifest)
    python_root=runtime/'public-python';payloads,checked=extract(archive,value['interpreter_sha256'],python_root)
    write_json(output/'python-payloads.json',payloads);write_json(output/'python-audit.json',checked)
    progress('interpreter')
    python=python_root/'python/bin/python3.12';venv=runtime/'venv'
    temporary=runtime/'temporary-install';temporary.mkdir(mode=0o700)
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(runtime),'PIP_CONFIG_FILE':'/dev/null',
         'PIP_NO_INDEX':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(temporary)}
    execute([*isolated_install_command(python,'venv',runtime,env),'--without-pip',str(venv)],env=env,check=True)
    install=output/'offline-install.txt'
    # Use the exact selected local wheels, including Torch's direct-URL lock row.
    # A direct URL in pip's original requirements would bypass --no-index.
    install.write_text(''.join(str(p.resolve())+' --hash=sha256:'+sha(p)+'\n' for p in sorted(wheels.glob('*.whl'))))
    execute([*isolated_install_command(python,'pip',runtime,env),'--python',str(venv/'bin/python'),'install','--no-index','--no-deps','--no-compile',
             '--require-hashes','-r',str(install)],env=env,check=True)
    progress('installation')
    installed=verify_installed(manifest,{'site':venv/'lib/python3.12/site-packages','prefix':venv,'scripts':venv/'bin','headers':venv/'include/python3.12'})
    write_json(output/'installed-audit.json',installed);progress('installed')
    inspected=launch(lock,wheels,venv,source/'src',output/'audited-inspection','ovl_pipeline.runtime_launch',['--inspect-current'],
                     interpreter_archive=archive,interpreter_sha256=value['interpreter_sha256'],interpreter_root=python_root,progress=progress,bytecode_root=runtime/'bytecode')
    progress('inspection')
    result={'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':expected,
            'bootstrap_executable_sha256':sha(Path('/proc/self/exe')),'wheel_manifest_sha256':digest(manifest),
            'python_manifest_sha256':digest(payloads),'installed_audit_sha256':digest(installed),'inspection':inspected,
            'scope':'complete public binary/package identity and constrained CPU startup; CUDA admission NOT_RUN',
            'target_executable_sha256':sha(python),'network_installation':'DISABLED','production_acceptance':'NOT_RUN'}
    write_json(output/'setup.json',result);return result


def audited(config_file,expected,inputs,runtime,output,module,arguments):
    runtime,output=layout(runtime,output)
    value,source,wheels,lock,manifest,archive=selected(config_file,expected,inputs,wheel_cache=runtime/'wheels')
    from ovl_pipeline.runtime_launch import launch
    return launch(lock,wheels,runtime/'venv',source/'src',output,module,arguments,
                  interpreter_archive=archive,interpreter_sha256=value['interpreter_sha256'],interpreter_root=runtime/'public-python',bytecode_root=runtime/'bytecode')


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
