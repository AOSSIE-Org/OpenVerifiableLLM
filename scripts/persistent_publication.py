"""One separately supervised local publication; never a GPU or trust admission.

Freeze inputs and the original absolute deadline before once-only service start.
An uncertain start is reconciled read-only. The existing publisher and the
receiving recorder remain responsible for every signature/download check.
"""
from pathlib import Path
import argparse
import fcntl
import os
import re
import subprocess
import sys
import threading
import time

# Isolated worker uses only the explicitly selected repository modules.
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer
from pod_job_client import save_once

PARAMETERS=('packet','registration-bundle','production-policy','source-policy','source-checkout','config',
            'chain-directory','previous-directory','previous-policies','output')


def path(value,*,python=False):
    if type(value) is not str or not re.fullmatch(r'/[A-Za-z0-9_.+/-]+',value):raise EvidenceError('closed absolute service path required')
    p=Path(value)
    if str(p)!=value or '..' in p.parts or any(q.is_symlink() for q in p.parents):raise EvidenceError('noncanonical or symlink service path')
    if not python and p.is_symlink():raise EvidenceError('service path symlink')
    return p


def tree(p):
    if p.is_file():return {'directory':str(p.parent),'files':inventory(p.parent,[p.name])}
    if not p.is_dir():raise EvidenceError('selected publisher input absent')
    names=[]
    for f in p.rglob('*'):
        if f.is_symlink() or not(f.is_file() or f.is_dir()):raise EvidenceError('unsafe publisher input tree')
        if f.is_file():names.append(f.relative_to(p).as_posix())
    return {'directory':str(p),'files':inventory(p,names)}


def sources(project):
    return sorted([p.relative_to(project).as_posix() for base in ('src','scripts')
                   for p in (project/base).rglob('*.py')]
                  +[p.relative_to(project).as_posix() for p in (project/'requirements').iterdir() if p.is_file()])


def stop_guard(stop_request):
    """Any observable request at the operator-selected path is binding.

    This read grants no liveness or verification credit. It cannot cancel a
    provider request that was already sent before the stop became observable.
    """
    p=path(str(stop_request))
    if p.exists():raise EvidenceError('original controller requests stop')


def selection(registration,boundary,deadline,arguments,*,python,stop_request=None):
    """Build an operator selection from already retained, independently checked inputs.

    Inventory construction is not trust validation. The worker's existing publisher
    verifies the selected policies, packet, chain and every public download.
    """
    fields(arguments,' '.join(PARAMETERS),'publisher arguments')
    value={'schema':'ovl.persistent-publication.v1','registration_sha256':registration,
           'boundary_sha256':boundary,'deadline_epoch':deadline,'project':str(ROOT),
           'python':str(path(str(python),python=True)),'python_sha256':file_hash(python),
           'arguments':arguments,'inputs':{name:tree(path(arguments[name])) for name in PARAMETERS[:-1]},
           'sources':inventory(ROOT,sources(ROOT))}
    if stop_request is not None:
        value['schema']='ovl.persistent-publication.v2'
        value['stop_request']=str(path(str(stop_request)))
    validate(value,digest(value))
    return value


def separate_writes(spec,directory):
    """Never place mutable service/publication output in a selected input tree."""
    outputs=[path(spec['arguments']['output']),path(str(directory))]
    inputs=[path(spec['arguments'][name]) for name in PARAMETERS[:-1]]
    for a in outputs:
        for b in inputs:
            if a==b or a.is_relative_to(b) or b.is_relative_to(a):
                raise EvidenceError('publisher output overlaps selected input')
    if outputs[0]==outputs[1] or any(a.is_relative_to(b) for a,b in (outputs,outputs[::-1])):
        raise EvidenceError('publisher state overlaps publication output')


def validate(spec,expected):
    v2=spec.get('schema')=='ovl.persistent-publication.v2'
    fields(spec,'schema registration_sha256 boundary_sha256 deadline_epoch project python python_sha256 arguments inputs sources'+(' stop_request' if v2 else ''),'persistent publication selection')
    if spec['schema'] not in ('ovl.persistent-publication.v1','ovl.persistent-publication.v2') or digest(spec)!=expected:raise EvidenceError('publication selection differs')
    if v2:path(spec['stop_request'])
    from ovl_pipeline.canonical import require_digest
    for k in ('registration_sha256','boundary_sha256','python_sha256'):require_digest(spec[k])
    integer(spec['deadline_epoch'],1,2**53-1,'original publisher deadline')
    project=path(spec['project']);python=path(spec['python'],python=True)
    if project!=ROOT or file_hash(python)!=spec['python_sha256']:raise EvidenceError('selected publisher runtime differs')
    fields(spec['arguments'],' '.join(PARAMETERS),'publisher arguments')
    fields(spec['inputs'],' '.join(PARAMETERS[:-1]),'publisher input inventories')
    for name,value in spec['arguments'].items():
        p=path(value)
        if name!='output' and tree(p)!=spec['inputs'][name]:raise EvidenceError('publisher input identity/completeness changed')
    # Select all executable project Python modules and dependency declarations.
    names=sources(project)
    if [f['path'] for f in spec['sources']]!=names:raise EvidenceError('publisher source selection incomplete')
    verify_inventory(project,spec['sources'])
    return project,python


def unit_text(spec,expected,directory,remaining):
    integer(remaining,1,86400,'bounded local publisher service lifetime')
    directory=path(str(Path(directory).absolute()));project,python=validate(spec,expected)
    # Closed paths and digest arguments contain no whitespace, expansions or
    # unit-specifier characters. No credentials are passed in the command.
    command=[str(python),'-I','-B',str(project/'scripts/persistent_publication.py'),'worker',
             '--selection',str(directory/'selection.json'),'--sha256',expected,'--state',str(directory)]
    return f'''[Unit]
Description=OpenVerifiableLLM selected public checkpoint publisher
[Service]
Type=exec
WorkingDirectory={project}
ExecStart={' '.join(command)}
Restart=no
RemainAfterExit=no
KillMode=control-group
SendSIGKILL=yes
TimeoutStartSec=20
TimeoutStopSec=5
RuntimeMaxSec={remaining}
MemoryMax=4G
StandardOutput=append:{directory}/stdout.log
StandardError=append:{directory}/stderr.log
'''


def command(argv):
    return subprocess.run(argv,capture_output=True,text=True,timeout=20,check=True).stdout


def observe(name,*,execute=command):
    raw=execute(['systemctl','--user','show',name,'--property=LoadState,ActiveState,SubState,MainPID,ExecMainCode,ExecMainStatus,Result,InvocationID,FragmentPath,DropInPaths'])
    result={}
    for line in raw.splitlines():
        k,sep,v=line.partition('=')
        if not sep or k in result:raise EvidenceError('invalid service observation')
        result[k]=v
    fields(result,'LoadState ActiveState SubState MainPID ExecMainCode ExecMainStatus Result InvocationID FragmentPath DropInPaths','publication service observation')
    return result


def start_or_adopt(spec,expected,directory,*,execute=command,wall=time.time,unit_directory=None):
    validate(spec,expected);directory=path(str(Path(directory).absolute()));separate_writes(spec,directory)
    if 'stop_request' in spec:stop_guard(spec['stop_request'])
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(directory/'.controller.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save_once(directory/'selection.json',spec)
        name='ovllm-publication-'+expected+'.service'
        units=Path(unit_directory) if unit_directory is not None else Path.home()/'.config/systemd/user'
        units=path(str(units.absolute()));units.mkdir(mode=0o700,parents=True,exist_ok=True);target=units/name
        fence=directory/'start-fence.json'
        if fence.exists():
            selected=read_json(fence)
            fields(selected,'schema selection_sha256 unit_file unit_sha256 runtime_seconds requested_epoch','publisher start fence')
            integer(selected['requested_epoch'],1,spec['deadline_epoch'],'original launch clock')
            integer(selected['runtime_seconds'],30,86400,'original service lifetime')
            if (selected['schema']!='ovl.publisher-start-fence.v1' or selected['selection_sha256']!=expected
                or selected['unit_file']!=str(target) or target.is_symlink() or file_hash(target)!=selected['unit_sha256']
                or selected['runtime_seconds']!=spec['deadline_epoch']-selected['requested_epoch']-30
                or (directory/'unit.service').is_symlink()
                or (directory/'unit.service').read_text()!=unit_text(spec,expected,directory,selected['runtime_seconds'])
                or (directory/'unit.service').read_bytes()!=target.read_bytes()):raise EvidenceError('original publisher service selection changed')
        else:
            now=int(wall());remaining=spec['deadline_epoch']-now-30
            if not 30<=remaining<=86400:raise EvidenceError('insufficient bounded publisher time before original deadline')
            text=unit_text(spec,expected,directory,remaining)
            if target.exists():raise EvidenceError('unfenced preexisting publisher unit; do not adopt or replace')
            for dest in (target,directory/'unit.service'):
                fd_write=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd_write,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
                parent_fd=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(parent_fd)
                finally:os.close(parent_fd)
            execute(['systemctl','--user','daemon-reload'])
            initial=observe(name,execute=execute)
            if initial['LoadState']!='loaded' or initial['FragmentPath']!=str(target) or initial['DropInPaths']:
                raise EvidenceError('publisher unit has foreign fragment or drop-ins')
            save_once(fence,{'schema':'ovl.publisher-start-fence.v1','selection_sha256':expected,'unit_file':str(target),
                'unit_sha256':file_hash(target),'runtime_seconds':remaining,'requested_epoch':now})
            # No caller, restart or timeout may repeat this start after the fence.
            if 'stop_request' in spec:stop_guard(spec['stop_request'])
            execute(['systemctl','--user','start',name])
        observation=observe(name,execute=execute)
        if observation['LoadState']!='loaded' or observation['FragmentPath']!=str(target) or observation['DropInPaths']:
            raise EvidenceError('publisher unit is absent or has a different owner-selected fragment')
        return {'schema':'ovl.publisher-service-observation.v1','selection_sha256':expected,'unit':name,
                'observation':observation,'scope':'owned local process observation only; publication and training verification separate'}
    finally:os.close(fd)


def worker(spec,expected,directory):
    validate(spec,expected);directory=path(str(Path(directory).absolute()));separate_writes(spec,directory)
    guard=None if 'stop_request' not in spec else lambda:stop_guard(spec['stop_request'])
    if guard is not None:guard()
    if read_json(directory/'selection.json')!=spec:raise EvidenceError('worker selection differs from original service')
    left=spec['deadline_epoch']-time.time()
    if left<=0:raise EvidenceError('original publisher deadline expired')
    fd=os.open(directory/'worker-started.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        from ovl_pipeline.canonical import canonical
        f.write(canonical({'selection_sha256':expected,'pid':os.getpid(),'started_epoch':int(time.time())}));f.flush();os.fsync(f.fileno())
    # Failure exits the main service; systemd then kills its entire control group.
    # Relative systemd runtime is an additional bound, never a deadline renewal.
    timer=threading.Timer(left,lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        from ovl_pipeline.anchoring import PublisherPolicy
        from ovl_pipeline.production_identity import ProductionPublisherPolicy
        from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
        from publish_progress_boundary import publish
        p={k:Path(v) for k,v in spec['arguments'].items()}
        result=publish(p['packet'],p['registration-bundle'],ProductionPublisherPolicy(**read_json(p['production-policy'])),
            PublisherPolicy(**read_json(p['source-policy'])),p['source-checkout'],read_json(p['config']),p['chain-directory'],
            p['previous-directory'],[ProgressPublisherPolicy(**v) for v in read_json(p['previous-policies'])],p['output'],spec['deadline_epoch'],
            **({} if guard is None else {'guard':guard}))
        validate(spec,expected)
        if guard is not None:guard()
        if (time.time()>=spec['deadline_epoch'] or result['registration_sha256']!=spec['registration_sha256']
            or result['boundary_sha256']!=spec['boundary_sha256'] or read_json(p['output']/'ack.json')!=result):
            raise EvidenceError('completed publisher identity/deadline differs')
        save_once(directory/'result.json',{'schema':'ovl.publisher-service-result.v1','selection_sha256':expected,
            'ack_sha256':digest(result),'ack_path':str(p['output']/'ack.json'),'scope':'publisher returned; caller must independently verify acknowledgement before delivery'})
        return result
    finally:timer.cancel()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','worker'])
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--state',type=Path,required=True);a=p.parse_args()
    try:
        result=(worker if a.action=='worker' else start_or_adopt)(read_json(a.selection),a.sha256,a.state)
        print('Selected publication process result '+digest(result))
    except Exception as error:p.exit(1,'persistent publication refused: '+type(error).__name__+'; preserve original service and output\n')


if __name__=='__main__':main()
