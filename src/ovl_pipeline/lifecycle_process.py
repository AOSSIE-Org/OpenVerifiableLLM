"""Reconciled local/remote workload processes, independent of transport sessions.

Only the caller-selected fixed module is executed, without a shell. A detached
supervisor persists the actual child exit and inventory. The same lease remains
open in supervisor and child, so losing either caller cannot permit overlap.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time

from .canonical import EvidenceError, canonical, digest, file_hash, read_json, require_digest, write_json
from .lifecycle import Pending, durable_mkdir, exclusive, sync_directory
from .lifecycle_artifacts import snapshot, check_snapshot, durable_tree

# These modules enforce their scientific input contracts. A process receipt is
# execution evidence only; it cannot replace their numerical/artifact verifiers.
MODULES = {
    'ovl_pipeline.lifecycle_fixture', 'ovl_pipeline.gpu_pilot',
    'ovl_pipeline.initialization', 'ovl_pipeline.production_record',
    'ovl_pipeline.production_replay', 'ovl_pipeline.production_export',
    'ovl_pipeline.production_verify', 'ovl_pipeline.release_download',
    'ovl_pipeline.runtime_launch',
}


def target(request):
    """Bind audited launch and numerical outputs to the selected workspace."""
    if request['module']!='ovl_pipeline.runtime_launch':
        return request['module'],Path(request['output']),None
    args=request['arguments']
    if args.count('--')!=1:
        raise EvidenceError('unambiguous audited workload required')
    split=args.index('--');outer,inner=args[:split],args[split+1:]
    if len(outer)%2 or len(set(outer[::2]))!=len(outer)//2:
        raise EvidenceError('invalid audited launch arguments')
    fields=dict(zip(outer[::2],outer[1::2]))
    required={'--lock','--wheels','--venv','--source','--output','--module',
              '--interpreter-archive','--interpreter-sha256','--interpreter-root'}
    if set(fields)-{'--allowed-generated'}!=required:
        raise EvidenceError('complete audited runtime and interpreter origin required')
    module=fields['--module']
    if module not in {'ovl_pipeline.lifecycle_fixture','ovl_pipeline.gpu_pilot','ovl_pipeline.initialization','ovl_pipeline.production_record',
                      'ovl_pipeline.production_replay','ovl_pipeline.production_export'}:
        raise EvidenceError('unsupported audited scientific target')
    if inner.count('--output')!=1 or inner.index('--output')==len(inner)-1:
        raise EvidenceError('unique scientific output required')
    numerical=Path(inner[inner.index('--output')+1]);audit=Path(fields['--output']);root=Path(request['output'])
    if (fields['--source']!=str(Path(request['source_root'])/'src')
            or numerical!=root/'result' or audit!=root/'audit'):
        raise EvidenceError('audited workload output/source binding differs')
    return module,numerical,audit


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid, 'start_ticks':stat[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'state':stat[0]}


def live(selected):
    try:
        now=process_identity(selected['pid'])
    except FileNotFoundError:
        return False
    # PermissionError is deliberately not converted into authentication failure,
    # absence or retry permission. No other process's /proc descriptors are read.
    return all(now[k]==selected[k] for k in ('pid','start_ticks','boot_id')) and now['state']!='Z'


def remaining(directory, request):
    """Keep the original elapsed-time bound through supervisor restarts."""
    bound=read_json(Path(directory)/'clock.json')
    if (set(bound)!={'operation','boot_id','stop_monotonic_ms'}
            or bound['operation']!=digest(request) or type(bound['stop_monotonic_ms']) is not int):
        raise EvidenceError('workload clock identity differs')
    if bound['boot_id']!=process_identity()['boot_id']:
        return 0  # Unknown elapsed lifetime after reboot never adds time.
    return min(request['deadline']-time.time(),bound['stop_monotonic_ms']/1000-time.monotonic())


def validate(request):
    if set(request) != {'schema','module','arguments','source_root','source_sha256','deadline','output','inputs'}:
        raise EvidenceError('invalid workload request')
    if request['schema']!='ovl.lifecycle-process.v1' or request['module'] not in MODULES:
        raise EvidenceError('unsupported workload module')
    require_digest(request['source_sha256'])
    if (type(request['arguments']) is not list or len(request['arguments'])>128
            or any(type(s) is not str or len(s.encode())>16384 or '\0' in s for s in request['arguments'])
            or type(request['deadline']) is not int or request['deadline']<=0):
        raise EvidenceError('invalid workload arguments/deadline')
    for name in ('source_root','output'):
        if type(request[name]) is not str or not Path(request[name]).is_absolute():
            raise EvidenceError('workload paths must be selected absolute paths')
    if type(request['inputs']) is not list or not request['inputs']:
        raise EvidenceError('workload requires frozen input files')
    names = set()
    for entry in request['inputs']:
        if set(entry) != {'path','bytes','sha256'} or entry['path'] in names:
            raise EvidenceError('invalid workload input inventory')
        names.add(entry['path'])
        path=Path(entry['path'])
        require_digest(entry['sha256'])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or type(entry['bytes']) is not int or path.stat().st_size!=entry['bytes']
                or file_hash(path)!=entry['sha256']):
            raise EvidenceError('workload input bytes differ')
    from .source_identity import code_root
    if request['source_sha256'] != code_root():
        raise EvidenceError('workload executing source differs')
    import ovl_pipeline
    expected=Path(request['source_root'])/'src/ovl_pipeline/__init__.py'
    if Path(ovl_pipeline.__file__).resolve()!=expected.resolve():
        raise EvidenceError('workload checkout import differs')
    target(request)
    if request['module']=='ovl_pipeline.runtime_launch':
        args=request['arguments'];split=args.index('--');fields=dict(zip(args[:split:2],args[1:split:2]))
        for key in ('--lock','--allowed-generated'):
            if key in fields and fields[key] not in names:
                raise EvidenceError('audited dependency selection file must be a frozen input: '+key)


def observe(directory, request):
    validate(request)
    directory=Path(directory)
    output=Path(request['output'])
    if directory==output or directory.is_relative_to(output) or output.is_relative_to(directory):
        raise EvidenceError('workload output and private process records must be separate')
    selected=digest(request)
    if not (directory/'request.json').exists():
        return {'status':'absent','operation':selected}
    if read_json(directory/'request.json')!=request:
        raise EvidenceError('workload request changed')
    terminal=directory/'terminal.json'
    if terminal.exists():
        result=read_json(terminal)
        if result['operation']!=selected:
            raise EvidenceError('workload terminal operation changed')
        if result['status']=='complete':
            check_snapshot(Path(request['output']),result['output'])
        return result
    identity=directory/'process.json'
    if identity.exists() and live(read_json(identity)):
        return {'status':'running','operation':selected}
    # Missing/racy process metadata does not establish absence of a held lease.
    try:
        with exclusive(directory/'lease'):
            return {'status':'uncertain','operation':selected}
    except Pending:
        return {'status':'running','operation':selected}


def spawn_supervisor(directory, request, lease, event):
    cache=Path(tempfile.mkdtemp(prefix='supervisor-cache-',dir=directory))
    env=parent_environment(request)
    with (directory/'supervisor.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-B','-s','-S','-P','-X','pycache_prefix='+str(cache),
                                '-m','ovl_pipeline.lifecycle_process','supervise',
                                '--directory',str(directory),'--operation',digest(request),'--lease',str(lease)],
                               env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,pass_fds=(lease,),start_new_session=True)
    event('submitted')
    return {'status':'submitted','operation':digest(request),'supervisor_pid':child.pid}


def parent_environment(request):
    env=dict(os.environ)
    for name in list(env):
        if name.startswith('PYTHON'):del env[name]
    prefix=Path(sys.executable).absolute().parent.parent
    # Python 3.12 -S skips venv prefix initialization. Preserve the installation
    # selected by this executable, rather than accidentally using its base site.
    site=(prefix/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages'
          if (prefix/'pyvenv.cfg').is_file() else Path(sysconfig.get_path('purelib')))
    env['PYTHONPATH']=os.pathsep.join((str(Path(request['source_root'])/'src'),str(site)))
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['TOKENIZERS_PARALLELISM']='false'
    return env


def submit(directory, request, *, event=lambda _:None):
    validate(request)
    directory=Path(directory)
    output=Path(request['output'])
    if directory==output or directory.is_relative_to(output) or output.is_relative_to(directory):
        raise EvidenceError('workload output and private process records must be separate')
    durable_mkdir(directory)
    with exclusive(directory/'lease') as lease:
        if (directory/'request.json').exists():
            raise Pending('original workload exists; observe or recover it, never duplicate submission')
        if time.time()>=request['deadline']:
            raise EvidenceError('workload deadline expired')
        output=Path(request['output'])
        if output.exists():
            raise EvidenceError('fresh workload output required')
        # Clock first: a crash before request publication may adopt only this
        # same original bound. It cannot create a request without its clock.
        if (directory/'clock.json').exists():
            if remaining(directory,request)<=0:
                raise EvidenceError('original workload deadline expired during submission')
        else:
            write_json(directory/'clock.json',{'operation':digest(request),'boot_id':process_identity()['boot_id'],
                       'stop_monotonic_ms':int((time.monotonic()+max(0,request['deadline']-time.time()))*1000)})
        write_json(directory/'request.json',request)
        event('intent')
        # A descriptor handoff closes the gap between launch and process receipt.
        # Do not infer work completion from this launch acknowledgement.
        return spawn_supervisor(directory,request,lease,event)


def recover(directory, request):
    """Recover local computation only after the original shared lease is free.

    Replay always starts afresh. Recording may use the scientific driver's
    validated resume entrypoint. No remote creation/publication mutation is here.
    """
    import uuid
    validate(request)
    directory=Path(directory)
    with exclusive(directory/'lease') as lease:
        if read_json(directory/'request.json') != request:
            raise EvidenceError('recovery request differs')
        terminal=directory/'terminal.json'
        if terminal.exists() and read_json(terminal).get('deadline_expired') is True:
            raise EvidenceError('recorded workload deadline failure cannot become completion')
        if terminal.exists() and read_json(terminal)['status']=='complete':
            result=read_json(terminal)
            if result['operation']!=digest(request):
                raise EvidenceError('workload terminal operation changed')
            check_snapshot(Path(request['output']),result['output'])
            return result
        module,numerical,audit=target(request)
        if audit is not None and (audit/'process.json').exists():
            receipt=read_json(audit/'process.json');launch=read_json(audit/'launch.json')
            args=request['arguments'];expected_args=args[args.index('--')+1:]
            prior_recovery=directory/'recovery.json'
            if prior_recovery.exists() and read_json(prior_recovery)['resume_recording'] and module=='ovl_pipeline.production_record':
                expected_args=[*expected_args,'--resume']
            if (receipt.get('schema')!='ovl.audited-runtime-process.v1'
                    or receipt.get('launch_sha256')!=digest(launch)
                    or launch['module']!=module or launch['arguments']!=expected_args
                    or launch['source']!=str(Path(request['source_root'])/'src')
                    or launch.get('lifecycle_binding')!={'request_sha256':digest(request),'source_sha256':request['source_sha256']}):
                raise EvidenceError('saved audited process completion differs')
            if receipt['exit_code']==0:
                durable_tree(Path(request['output']));sync_directory(Path(request['output']).parent)
                result={'operation':digest(request),'status':'complete','exit_code':0,'deadline_expired':None,
                        'scope':'reconciled audited target exit and bytes; original supervisor exit unknown; scientific validation required',
                        'audited_process_sha256':digest(receipt),'output':snapshot(Path(request['output']))}
                write_json(terminal,result)
                return result
        if module=='ovl_pipeline.production_record' and (numerical/'record.json').exists():
            raise Pending('completed recording without audited exit requires reconciliation; never resume completed recording')
        if remaining(directory,request) <= 0:
            raise EvidenceError('recovery cannot extend original workload deadline')
        recovery=directory/'recovery'/uuid.uuid4().hex
        durable_mkdir(recovery)
        for name in ('process.json','child.json','terminal.json','stdout.log','stderr.log','supervisor.log'):
            path=directory/name
            if path.exists():
                with path.open('rb') as f: os.fsync(f.fileno())
                path.rename(recovery/name)
        output=Path(request['output'])
        module,numerical,audit=target(request)
        resume=False
        if output.exists():
            if request['module']=='ovl_pipeline.lifecycle_fixture' and request['arguments'][0]=='run':
                resume=True  # Its private journal authenticates every completed stage.
            elif module=='ovl_pipeline.production_record' and (numerical/'chain.json').is_file():
                resume=True  # Driver verifies full state and public prefix first.
                if audit is not None and audit.exists():
                    durable_tree(audit)
                    audit.rename(recovery/'prior-audit')
                    sync_directory(audit.parent)
            else:
                durable_tree(output)
                output.rename(recovery/'partial-output')
                sync_directory(output.parent)
        sync_directory(recovery)
        sync_directory(directory)
        # Preserve the previous resume interpretation until its replacement is
        # atomically durable. A crash may otherwise misinterpret the old audit.
        if (directory/'recovery.json').exists():
            write_json(recovery/'recovery.json',read_json(directory/'recovery.json'))
        write_json(directory/'recovery.json',{'operation':digest(request),'resume_recording':resume,
                                             'prior_attempt':recovery.name,'replay_from_prover_state':False})
        return spawn_supervisor(directory,request,lease,lambda _:None)


def supervise(directory, operation, lease):
    directory=Path(directory)
    request=read_json(directory/'request.json')
    validate(request)
    if digest(request)!=operation:
        raise EvidenceError('supervisor request pin differs')
    stat=os.fstat(lease)
    expected=(directory/'lease/owner.lock').stat()
    if (stat.st_dev,stat.st_ino)!=(expected.st_dev,expected.st_ino):
        raise EvidenceError('supervisor lease descriptor differs')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(directory/'process.json',process_identity())
    env=parent_environment(request)
    command=[sys.executable,'-m',request['module'],*request['arguments']]
    module,_,_=target(request)
    if request['module']=='ovl_pipeline.runtime_launch':
        command[3:3]=['--lifecycle-lease-fd',str(lease),'--lifecycle-lease-path',str(directory/'lease/owner.lock'),
                      '--lifecycle-request',str(directory/'request.json')]
    recovery=directory/'recovery.json'
    if recovery.exists():
        chosen=read_json(recovery)
        if chosen['operation']!=operation:
            raise EvidenceError('recovery operation differs')
        if chosen['resume_recording'] and module=='ovl_pipeline.production_record' and '--resume' not in command:
            command.append('--resume')
    # Bind the actual child to this enforcing supervisor. The audited bootstrap
    # repeats that binding for its numerical child, including startup races.
    runner=env.pop('OVL_WORKLOAD_NUMERIC_RUNNER',None)
    if runner and request['module']=='ovl_pipeline.runtime_launch':
        raise EvidenceError('audited runtime requires native execution without an injected numeric runner')
    from .process_safety import identity
    wrapper=directory/'numeric-entry.py'
    wrapper.write_text("import runpy\nfrom ovl_pipeline.process_safety import bind_parent\nif __name__ == '__main__':\n    bind_parent("+repr(identity(os.getpid()))+")\n    runpy.run_module("+repr(request['module'])+",run_name='__main__')\n")
    cache=Path(tempfile.mkdtemp(prefix='wrapper-cache-',dir=directory))
    native=[sys.executable,'-B','-s','-S','-P','-X','pycache_prefix='+str(cache)]
    command=([runner,'run','--python',sys.executable] if runner else native)+[str(wrapper),*command[3:]]
    deadline=time.monotonic()+max(0,remaining(directory,request))
    if remaining(directory,request)<=0:
        result={'operation':operation,'status':'failed','exit_code':None,'deadline_expired':True,
                'scope':'not-started; original deadline expired','output':None}
        write_json(directory/'terminal.json',result)
        os.close(lease)
        return result
    with (directory/'stdout.log').open('ab') as stdout,(directory/'stderr.log').open('ab') as stderr:
        child=subprocess.Popen(command,cwd=request['source_root'],env=env,stdin=subprocess.DEVNULL,
                               stdout=stdout,stderr=stderr,pass_fds=(lease,),start_new_session=True)
        write_json(directory/'child.json',process_identity(child.pid))
        expired=False
        while child.poll() is None:
            if time.time()>=request['deadline'] or time.monotonic()>=deadline:
                expired=True
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                break
            time.sleep(min(0.25,max(0.01,deadline-time.monotonic())))
        # A suspended supervisor may first observe an exit after the bound.
        # Without timely exit evidence, do not affirm completion within it.
        expired=expired or time.time()>=request['deadline'] or time.monotonic()>=deadline
    result={'operation':operation,'status':'failed','exit_code':child.returncode,'deadline_expired':expired,
            'scope':'actual-child-exit-and-byte-inventory-only; scientific validation required','output':None}
    if not expired and child.returncode==0:
        durable_tree(Path(request['output']))
        sync_directory(Path(request['output']).parent)
        result['output']=snapshot(Path(request['output']))
        result['status']='complete'
    for name in ('stdout.log','stderr.log'):
        with (directory/name).open('rb') as f:os.fsync(f.fileno())
    write_json(directory/'terminal.json',result)
    os.close(lease)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('submit','observe','recover','supervise'))
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--operation')
    parser.add_argument('--lease',type=int)
    args=parser.parse_args()
    if args.action=='supervise': result=supervise(args.directory,args.operation,args.lease)
    else:
        if args.request is None:parser.error('--request required')
        request=read_json(args.request)
        result={'submit':submit,'observe':observe,'recover':recover}[args.action](args.directory,request)
    print(canonical(result).decode())


if __name__=='__main__':main()
