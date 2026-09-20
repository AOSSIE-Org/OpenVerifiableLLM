#!/usr/bin/env python3
"""One-shot detached workload runner, driven only by an operator-pinned job hash.

Standard library only, so it can bootstrap a separately audited runtime. This is
process/cost supervision, not model verification or provider attestation. It never
accepts an unpinned command from a model artifact, creates compute or retries a job.
The external host still owns provider teardown even if this process dies.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


class Refusal(RuntimeError):pass


ENVIRONMENT={'PATH','HOME','LANG','LC_ALL','PYTHONPATH','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','OVL_ACTIVITY_FILE','XDG_CACHE_HOME','TMPDIR',
             'UV_CACHE_DIR','HF_HUB_DISABLE_TELEMETRY','HF_HUB_DISABLE_PROGRESS_BARS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'}


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def regular(path):
    path=Path(path)
    for current in [path,*path.parents]:
        if current.is_symlink():raise Refusal('symlink path')
    if not stat.S_ISREG(path.stat().st_mode):raise Refusal('regular file required')
    return path


def object_file(path):
    path=regular(path)
    if not 0<path.stat().st_size<=16*1024**2:raise Refusal('JSON size')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise Refusal('duplicate JSON key')
            d[k]=v
        return d
    def constant(value):raise Refusal('nonfinite JSON')
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=constant)


def save(path,value,*,exclusive=False):
    path=Path(path);data=encoded(value)
    fd,name=tempfile.mkstemp(prefix='.job-pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if path.is_symlink():raise Refusal('output symlink')
        if exclusive:os.link(name,path)
        else:os.replace(name,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(name):os.unlink(name)


def load(directory,expected,worker_sha256,*,resolve_executable=True):
    directory=Path(directory)
    if not directory.is_absolute() or '..' in directory.parts:raise Refusal('absolute owned job directory required')
    if not re.fullmatch('[0-9a-f]{64}',expected) or not re.fullmatch('[0-9a-f]{64}',worker_sha256):raise Refusal('explicit code/job hashes required')
    if hash_file(regular(__file__))!=worker_sha256:raise Refusal('worker code differs from operator selection')
    path=regular(directory/'job.json')
    if hash_file(path)!=expected:raise Refusal('job differs from operator selection')
    v=object_file(path)
    return validate_job(v,resolve_executable=resolve_executable)


def validate_job(v,*,resolve_executable=True):
    """Pure descriptor checks also used by the off-pod preflight before rental."""
    names={'schema','kind','argv','cwd','environment','deadline_epoch','stop_grace_seconds','minimum_free_bytes','required_files','export_roots'}
    if type(v) is not dict or set(v)!=names or v['schema']!='ovl.pod-job.v1':raise Refusal('job schema')
    if v['kind'] not in ('setup','pilot','production-record','full-replay','export'):raise Refusal('job kind')
    argv=v['argv']
    if (type(argv) is not list or not 1<=len(argv)<=256 or
        any(type(a) is not str or not a or '\x00' in a or len(a)>8192 for a in argv) or not argv[0].startswith('/')):
        raise Refusal('explicit bounded argv required; no shell interpretation')
    if type(v['cwd']) is not str or not Path(v['cwd']).is_absolute() or '..' in Path(v['cwd']).parts:
        raise Refusal('explicit working directory required')
    if type(v['environment']) is not dict or not set(v['environment'])<=ENVIRONMENT:
        raise Refusal('unapproved or credential environment')
    if any(type(k) is not str or type(x) is not str or '\x00' in x or len(x)>8192 for k,x in v['environment'].items()):
        raise Refusal('invalid environment')
    for k,lo,hi in [('deadline_epoch',1,2**53-1),('stop_grace_seconds',1,300),('minimum_free_bytes',1,2**50)]:
        if type(v[k]) is not int or not lo<=v[k]<=hi:raise Refusal('invalid job bound')
    files=v['required_files']
    if type(files) is not list or not files or len(files)>100000:raise Refusal('required input inventory')
    seen=set()
    for f in files:
        if (type(f) is not dict or set(f)!={'path','bytes','sha256'} or type(f['path']) is not str
            or not Path(f['path']).is_absolute() or '..' in Path(f['path']).parts or f['path'] in seen
            or type(f['bytes']) is not int or not 0<=f['bytes']<=2**40 or not re.fullmatch('[0-9a-f]{64}',f['sha256'])):
            raise Refusal('required input shape')
        seen.add(f['path'])
    roots=v['export_roots']
    if type(roots) is not list or not roots or len(roots)>32 or len(set(roots))!=len(roots):raise Refusal('explicit export roots required')
    for root in roots:
        if type(root) is not str or not Path(root).is_absolute() or '..' in Path(root).parts:raise Refusal('export root shape')
    if resolve_executable and v['kind']!='setup' and not any(f['path']==str(Path(v['argv'][0]).resolve(strict=True)) for f in files):
        raise Refusal('resolved workload executable must be hash bound')
    return v


def process_identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text();tail=text[text.rfind(')')+2:].split()
    return {'pid':pid,'start_ticks':int(tail[19]),'process_group':int(tail[2])}


def signal_owned(identity,sig):
    try:current=process_identity(identity['pid'])
    except FileNotFoundError:return False
    if current!=identity or current['process_group']!=current['pid']:raise Refusal('process identity changed; refusing signal')
    try:os.killpg(current['pid'],sig)
    except ProcessLookupError:return False
    return True


def exit_ready(pid):
    # WNOWAIT retains the leader's identity until descendants in its process
    # group have received termination, avoiding a reap/PID-reuse signal race.
    # https://docs.python.org/3.12/library/os.html#os.waitid
    return os.waitid(os.P_PID,pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None


def alive(identity):
    if identity is None:return False
    try:
        if process_identity(identity['pid'])!=identity:return False
        text=Path(f'/proc/{identity["pid"]}/stat').read_text()
        return text[text.rfind(')')+2:].split()[0] not in ('Z','X')
    # A task may disappear after /proc was opened but before read(). Linux
    # then returns ESRCH rather than ENOENT. Both observations mean absent;
    # permission, malformed metadata and signaling identity failures stay strict.
    except (FileNotFoundError,ProcessLookupError):return False


def metadata(directory,name,expected):
    path=Path(directory)/name
    if not path.exists():return None
    value=object_file(path)
    if value.get('job_sha256')!=expected:raise Refusal('foreign job metadata')
    return value


def supervision(directory,expected,worker_sha256):
    """Read-only identity/liveness observation; never restarts an absent runner."""
    directory=Path(directory);load(directory,expected,worker_sha256,resolve_executable=False)
    terminal=metadata(directory,'exit.json',expected) or metadata(directory,'abandoned.json',expected)
    receipt=metadata(directory,'launch/receipt.json',expected);child=metadata(directory,'launch/child.json',expected)
    runner=receipt['runner'] if receipt else None;process=child['process'] if child else None
    held=False;lock=directory/'launch/worker.lock'
    if lock.exists():
        fd=os.open(regular(lock),os.O_RDWR|os.O_NOFOLLOW)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:held=True
        finally:os.close(fd)
    if terminal is not None:state=terminal['state']
    elif held:state='RUNNING_UNVERIFIED'
    elif alive(runner):state='STARTING_UNVERIFIED'
    elif (directory/'launch/execution-intent.json').exists() and child is None:state='CHILD_IDENTITY_UNKNOWN'
    elif not(directory/'launch/intent.json').exists():state='LAUNCH_FENCE_WITHOUT_INTENT' if (directory/'launch').exists() else 'LAUNCH_NOT_OBSERVED'
    else:state='SUPERVISOR_ABSENT'
    return {'schema':'ovl.pod-job-supervision.v1','job_sha256':expected,'state':state,'runner_alive':alive(runner),
            'child_alive':alive(process),'terminal':terminal,'scope':'owned process observations only; no numerical verification'}


def abandon(directory,expected,worker_sha256):
    """Stop a recorded orphan after acquiring its absent supervisor's lease.

    No process identity is guessed. A spawn without a child receipt is unresolved
    and needs the external provider guard; it cannot receive a terminal record.
    """
    directory=Path(directory);load(directory,expected,worker_sha256,resolve_executable=False);launch=directory/'launch'
    if not launch.is_dir():raise Refusal('launch not observed; do not restart or invent a terminal status')
    fd=os.open(launch/'worker.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal=metadata(directory,'exit.json',expected) or metadata(directory,'abandoned.json',expected)
        if terminal is not None:return terminal
        receipt=metadata(directory,'launch/receipt.json',expected)
        if receipt is not None and alive(receipt['runner']):raise Refusal('supervisor is alive; cannot abandon')
        child=metadata(directory,'launch/child.json',expected)
        if child is None and (launch/'execution-intent.json').exists():raise Refusal('spawn identity unknown; external teardown required')
        identity=child['process'] if child else None
        if identity is not None and alive(identity):
            signal_owned(identity,signal.SIGKILL)
            deadline=time.monotonic()+5
            while alive(identity) and time.monotonic()<deadline:time.sleep(.05)
            if alive(identity):raise Refusal('recorded child still alive; no terminal acknowledgement')
        value={'schema':'ovl.workload-job-abandonment.v1','job_sha256':expected,'state':'ABANDONED',
               'observed_child':identity,'exit_code':'UNAVAILABLE',
               'scope':'supervisor absent; recorded child stopped or absent; no successful computation claim'}
        save(directory/'abandoned.json',value,exclusive=True);return value
    finally:os.close(fd)


def start(directory,expected,worker_sha256):
    directory=Path(directory);v=load(directory,expected,worker_sha256)
    if time.time()>=v['deadline_epoch']:raise Refusal('job deadline already passed')
    launch=directory/'launch'
    # mkdir is the creation fence. A missing receipt after this point is
    # ambiguous and may only be inspected/stopped; never spawn again.
    launch.mkdir(mode=0o700,exist_ok=False)
    save(launch/'intent.json',{'schema':'ovl.pod-job-launch-intent.v1','job_sha256':expected,'worker_sha256':worker_sha256},exclusive=True)
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run',str(directory),expected,worker_sha256],
                       stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                       start_new_session=True,close_fds=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
    receipt={'schema':'ovl.pod-job-launch.v1','job_sha256':expected,'worker_sha256':worker_sha256,'runner':process_identity(p.pid)}
    save(launch/'receipt.json',receipt,exclusive=True)
    return receipt


def run(directory,expected,worker_sha256):
    directory=Path(directory);v=load(directory,expected,worker_sha256);launch=directory/'launch'
    expected_intent={'schema':'ovl.pod-job-launch-intent.v1','job_sha256':expected,'worker_sha256':worker_sha256}
    if object_file(launch/'intent.json')!=expected_intent:raise Refusal('launch intent differs')
    fd=os.open(launch/'worker.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    child=None;identity=None;owns_execution=False
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:os.close(fd);raise
    try:
        if (directory/'abandoned.json').exists():raise Refusal('job was explicitly abandoned; never execute')
        save(launch/'execution-intent.json',expected_intent,exclusive=True)
        owns_execution=True
        started=time.monotonic();remaining=max(0,v['deadline_epoch']-time.time());deadline=started+remaining
        checked_bytes=0
        for f in v['required_files']:
            if time.time()>=v['deadline_epoch'] or time.monotonic()>=deadline:raise Refusal('deadline during input verification')
            p=regular(f['path'])
            if p.stat().st_size!=f['bytes'] or hash_file(p)!=f['sha256']:raise Refusal('required input differs')
            checked_bytes+=f['bytes']
            save(directory/'input-progress.json',{'schema':'ovl.job-input-progress.v1','job_sha256':expected,'verified_bytes':checked_bytes})
        if shutil.disk_usage(directory).free<v['minimum_free_bytes']:raise Refusal('insufficient free space')
        # Environment is explicitly selected; no HF/RunPod/signing credentials,
        # agent sockets, loader overrides or inherited Python hooks enter the job.
        with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(v['argv'],cwd=v['cwd'],env=v['environment'],stdin=subprocess.DEVNULL,
                                   stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
            identity=process_identity(child.pid)
            save(launch/'child.json',{'schema':'ovl.pod-job-child.v1','job_sha256':expected,'process':identity},exclusive=True)
            stop_at=None;reason=None
            while not exit_ready(child.pid):
                now=time.monotonic()
                if now>=deadline or time.time()>=v['deadline_epoch']:reason='job-deadline';stop_at=now
                elif (directory/'request-stop').exists() and stop_at is None:reason='operator-stop';stop_at=now+v['stop_grace_seconds']
                elif ((directory/'stdout.log').stat().st_size+(directory/'stderr.log').stat().st_size>16*1024**2
                      or shutil.disk_usage(directory).free<v['minimum_free_bytes']):reason='storage-bound';stop_at=now
                if stop_at is not None and now>=stop_at:
                    signal_owned(identity,signal.SIGTERM)
                    term_deadline=time.monotonic()+2
                    while not exit_ready(child.pid) and time.monotonic()<term_deadline:time.sleep(.05)
                    break
                save(directory/'status.json',{'schema':'ovl.pod-job-status.v1','job_sha256':expected,'state':'RUNNING',
                                             'process':identity,'stop_reason':reason})
                time.sleep(min(1,max(0.01,deadline-time.monotonic())))
            # Even a normally exiting leader must not leave background group
            # members writing into an allegedly final export. The selected
            # trusted workload must not deliberately escape into another session.
            signal_owned(identity,signal.SIGKILL)
            code=child.wait(timeout=2)
        status={'schema':'ovl.workload-job-exit.v1','job_sha256':expected,'state':'EXITED','exit_code':code}
        save(directory/'exit.json',status,exclusive=True)
        save(directory/'status.json',{'schema':'ovl.pod-job-status.v1','job_sha256':expected,'state':'EXITED',
                                     'process':identity,'stop_reason':reason})
        return status
    except BaseException as error:
        if child is not None and identity is not None:
            try:signal_owned(identity,signal.SIGKILL);child.wait(timeout=2)
            except Exception:pass
        if owns_execution:
            save(directory/'failure.json',{'schema':'ovl.pod-job-failure.v1','job_sha256':expected,'exception_type':type(error).__name__,
                                          'retry':'FORBIDDEN; inspect retained launch/child state and export before provider teardown'},exclusive=True)
            if child is None or child.poll() is not None:
                save(directory/'exit.json',{'schema':'ovl.workload-job-exit.v1','job_sha256':expected,'state':'EXITED',
                                            'exit_code':125 if child is None else child.returncode},exclusive=True)
        raise
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['start','run','inspect','abandon'])
    p.add_argument('directory',type=Path);p.add_argument('job_sha256');p.add_argument('worker_sha256');a=p.parse_args()
    try:
        result={'start':start,'run':run,'inspect':supervision,'abandon':abandon}[a.operation](a.directory,a.job_sha256,a.worker_sha256)
        print(encoded(result).decode())
    except Exception as error:p.exit(1,'job refused: '+type(error).__name__+'; retain journal, no automatic retry\n')


if __name__=='__main__':main()
