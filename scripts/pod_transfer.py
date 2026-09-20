#!/usr/bin/env python3
"""Bounded SSH file transport for an already adopted pod; never provisioning.

The caller independently selects the pod/endpoint observation and host-key pin.
Host-key TOFU is not provider attestation. Actual byte transfers can supply cost
progress; process liveness and remote clocks cannot. Private keys are local only.
"""
from pathlib import Path,PurePosixPath
import hashlib
import ipaddress
import os
import re
import selectors
import shlex
import signal
import stat
import subprocess
import time

from ovl_pipeline.canonical import EvidenceError,confined,file_hash,parse_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer


class TransientTransportError(EvidenceError):
    """Failed transport only; no identity, framing or content verification credit."""


def process_failure(code, errors):
    # Diagnostics stay private; only this closed category may be retried by a
    # read-only caller. Authentication/host-key/unknown failures remain fatal.
    # Every nonempty line must be a selected transport diagnostic. A transient
    # substring must never hide another line reporting a key-loading, signing,
    # authentication, host-identity or unknown failure. Stderr is not attestation
    # of its producer: this grants only a bounded read retry, never acceptance.
    patterns = (
        rb'(?:ssh: connect to host [0-9.]+ port [0-9]+: )?(?:connection timed out|connection refused|network is unreachable|no route to host)',
        rb'(?:(?:kex|ssh)_exchange_identification: (?:read: )?)?connection (?:reset(?: by peer)?|closed(?: by remote host)?)',
        rb'connection (?:reset|closed) by [0-9.]+ port [0-9]+',
        rb'connection to [0-9.]+ closed by remote host\.',
    )
    lines = [line.strip().lower() for line in bytes(errors).splitlines() if line.strip()]
    if code == 255 and lines and all(any(re.fullmatch(p,line) for p in patterns) for line in lines):
        return TransientTransportError('SSH connection failed; diagnostics withheld')
    return EvidenceError('SSH transfer process failed; diagnostics withheld')


def fatal_diagnostic(errors):
    # Act on complete received denial lines without waiting for process exit.
    # Incomplete chunks remain buffered; neither text nor key paths are exposed.
    denied=(b'host key',b'host identification',b'permission denied',b'authentication',
            b'load key ',b'sign_and_send_pubkey:',b'no more authentication methods')
    return any(any(word in line.lower() for word in denied) for line in bytes(errors).split(b'\n')[:-1])


def deadline_failure(process,errors):
    # A timed-out process may already have reported a strict failure. Do not
    # erase that evidence merely because it has not exited. Empty stderr gives
    # no identity/verification credit, but allows the original finite read retry.
    # One bounded nonblocking drain includes diagnostics already queued at the
    # observation boundary. No waiting, deadline renewal or acceptance occurs.
    if not process.stderr.closed:
        while len(errors)<=65536:
            try:part=os.read(process.stderr.fileno(),65537-len(errors))
            except BlockingIOError:break
            if not part:break
            errors.extend(part)
        if len(errors)>65536:return EvidenceError('SSH diagnostics exceeded bound')
    code=process.poll()
    if code is not None:
        if code==0:return EvidenceError('SSH process completed outside transfer deadline; diagnostics withheld')
        return process_failure(code,errors)
    if errors:return process_failure(255,errors)
    return TransientTransportError('transfer deadline reached; rental deadline is unchanged')

# This receives only a fixed root, confined relative path, length/hash and an
# explicit replacement bit. It drains and verifies the complete input before an
# atomic installation. It never evaluates a command from a downloaded artifact.
REMOTE_PUT=r'''
import hashlib,json,os,stat,sys,tempfile
from pathlib import Path
root,name,size,expected,replace=sys.argv[1:];size=int(size)
if not root.startswith('/') or '..' in Path(root).parts or str(Path(root))!=root or any(x in ('','.','..') for x in name.split('/')):raise ValueError('path')
base=Path(root);dest=base/name
current=Path('/')
for part in dest.parent.parts[1:]:
 current=current/part
 if current.is_symlink():raise ValueError('symlink parent')
 current.mkdir(mode=0o700,exist_ok=True)
 if not current.is_dir():raise ValueError('parent')
fd,temporary=tempfile.mkstemp(prefix='.ovllm-transfer-',dir=dest.parent)
try:
 h=hashlib.sha256();count=0
 with os.fdopen(fd,'wb') as f:
  while True:
   data=sys.stdin.buffer.read(min(1024*1024,size-count+1))
   if not data:break
   count+=len(data)
   if count>size:raise ValueError('oversize')
   h.update(data);f.write(data)
  f.flush();os.fsync(f.fileno())
 if count!=size or h.hexdigest()!=expected:raise ValueError('content')
 if dest.is_symlink():raise ValueError('symlink destination')
 if dest.exists():
  if not dest.is_file():raise ValueError('destination')
  old=hashlib.sha256()
  with dest.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):old.update(b)
  if dest.stat().st_size==size and old.hexdigest()==expected:os.unlink(temporary)
  elif replace=='1' and name in ('external-progress-policies.json','request-stop'):os.replace(temporary,dest)
  else:raise ValueError('immutable destination differs')
 else:
  os.link(temporary,dest);os.unlink(temporary)
 dfd=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(dfd)
 finally:os.close(dfd)
 print(json.dumps({'bytes':size,'sha256':expected},sort_keys=True,separators=(',',':')))
finally:
 # Sender retains the complete source. Temporary transfer duplicates alone may
 # be removed; no recorded checkpoint or sole evidence copy is deleted.
 if os.path.exists(temporary):os.unlink(temporary)
'''


REMOTE_GET=r'''
import base64,hashlib,json,os,stat,sys
from pathlib import Path
root,name,size,action=sys.argv[1:];size=int(size)
if action not in ('get','describe','observe'):raise ValueError('action')
if not root.startswith('/') or '..' in Path(root).parts or str(Path(root))!=root or any(x in ('','.','..') for x in name.split('/')):raise ValueError('path')
path=Path(root)/name;current=Path('/')
for part in path.parts[1:]:
 current=current/part
 if current.is_symlink():raise ValueError('symlink path')
if action in ('describe','observe') and not path.exists():
 print(json.dumps({'path':name,'present':False},sort_keys=True,separators=(',',':')));raise SystemExit(0)
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
with os.fdopen(fd,'rb') as f:
 info=os.fstat(f.fileno())
 if not stat.S_ISREG(info.st_mode) or not 0<=info.st_size<=size or action=='get' and info.st_size!=size:raise ValueError('file length/type')
 count=0;h=hashlib.sha256();observed=bytearray()
 while True:
  data=f.read(min(1024*1024,size-count+1))
  if not data:break
  count+=len(data)
  if count>size:raise ValueError('file grew')
  h.update(data)
  if action=='get':sys.stdout.buffer.write(data)
  elif action=='observe':observed.extend(data)
 if count!=info.st_size:raise ValueError('file length changed')
 if action=='describe':print(json.dumps({'path':name,'present':True,'bytes':count,'sha256':h.hexdigest()},sort_keys=True,separators=(',',':')))
 if action=='observe':print(json.dumps({'path':name,'present':True,'bytes':count,'sha256':h.hexdigest(),'data_b64':base64.b64encode(observed).decode()},sort_keys=True,separators=(',',':')))
 sys.stdout.buffer.flush()
'''


def private_file(path,*,maximum=None):
    path=Path(path)
    if path.is_symlink():raise EvidenceError('local SSH files must not be symlinks')
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
        raise EvidenceError('local SSH files must be owner-only regular files')
    if maximum is not None and info.st_size>maximum:raise EvidenceError('local SSH file exceeds bound')
    return path.resolve(strict=True)


def validate(profile,key,known_hosts):
    fields(profile,'schema pod_id host port user remote_root endpoint_observation_sha256 known_hosts_sha256 host_key_trust','SSH profile')
    if profile['schema']!='ovl.pod-ssh-profile.v1' or profile['user']!='root':raise EvidenceError('unsupported SSH profile')
    if type(profile['pod_id']) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,96}',profile['pod_id']):raise EvidenceError('invalid adopted pod identity')
    try:ipaddress.IPv4Address(profile['host'])
    except Exception:raise EvidenceError('explicit provider-observed IPv4 address required') from None
    integer(profile['port'],1,65535,'SSH endpoint port')
    if type(profile['remote_root']) is not str or not re.fullmatch('/workspace/ovllm/[a-z0-9][a-z0-9-]{0,127}',profile['remote_root']):
        raise EvidenceError('closed task-owned remote root required')
    if profile['host_key_trust'] not in ('operator-pinned-TOFU','operator-pinned-independently-corroborated'):
        raise EvidenceError('explicit host-key trust scope required')
    require_digest(profile['endpoint_observation_sha256']);require_digest(profile['known_hosts_sha256'])
    private_file(key,maximum=65536);known_hosts=private_file(known_hosts,maximum=65536)
    if file_hash(known_hosts)!=profile['known_hosts_sha256']:raise EvidenceError('SSH host-key file differs from external selection')
    # OpenSSH performs actual key matching. No global/user ssh config or agent
    # may override the selected endpoint or forward credentials.
    return profile


def relative(name):
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*',name) or any(x in ('.','..') for x in name.split('/')):
        raise EvidenceError('unsafe remote relative path')
    if str(PurePosixPath(name))!=name:raise EvidenceError('noncanonical remote path')
    return name


class Transport:
    def __init__(self,profile,key,known_hosts,*,popen=subprocess.Popen,wall=time.time,monotonic=time.monotonic):
        self.profile=validate(profile,key,known_hosts);self.key=Path(key);self.known=Path(known_hosts)
        self.popen=popen;self.wall=wall;self.monotonic=monotonic

    def command(self,remote_argv):
        validate(self.profile,self.key,self.known)
        # Quiet mode also suppresses the errors used by process_failure. Keep
        # error diagnostics in the bounded private stderr buffer; never print
        # them or broaden the closed retry classification.
        return ['ssh','-F','/dev/null','-T','-o','LogLevel=ERROR','-i',str(self.key.resolve()),'-p',str(self.profile['port']),
            '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','ForwardAgent=no',
            '-o','ClearAllForwardings=yes','-o','ControlMaster=no','-o','ControlPath=none',
            '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(self.known.resolve()),
            '-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
            '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','root@'+self.profile['host'],
            'exec '+shlex.join(remote_argv)]

    def stream(self,argv,destination,maximum,deadline,*,source=None,source_bytes=None,progress=None):
        integer(maximum,0,2**40,'transfer output bound');integer(deadline,1,2**53-1,'transfer deadline')
        if source is not None:integer(source_bytes,0,2**40,'transfer input bound')
        remaining=deadline-self.wall()
        if remaining<=0:raise EvidenceError('transfer deadline expired')
        end=self.monotonic()+remaining;out_count=in_count=0;errors=bytearray();pending=b'';input_done=source is None
        process=self.popen(self.command(argv),stdin=subprocess.DEVNULL if source is None else subprocess.PIPE,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,start_new_session=True)
        selector=None
        try:
            selector=selectors.DefaultSelector()
            for stream,kind in [(process.stdout,'output'),(process.stderr,'error')]:
                os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,kind)
            if source is not None:
                os.set_blocking(process.stdin.fileno(),False);selector.register(process.stdin,selectors.EVENT_WRITE,'input')
            last_observed=self.monotonic();last_counts=(0,0)
            while selector.get_map():
                left=min(end-self.monotonic(),deadline-self.wall())
                if left<=0:raise deadline_failure(process,errors)
                for key,events in selector.select(min(1,left)):
                    channel=key.fileobj
                    if key.data=='input':
                        if not pending:pending=source.read(min(1024*1024,source_bytes-in_count+1))
                        if len(pending)+in_count>source_bytes:raise EvidenceError('SSH transfer source exceeded selected bound')
                        if not pending:
                            selector.unregister(channel);channel.close();input_done=True;continue
                        try:n=os.write(channel.fileno(),pending)
                        except BlockingIOError:continue
                        pending=pending[n:];in_count+=n
                    else:
                        try:data=os.read(channel.fileno(),1024*1024)
                        except BlockingIOError:continue
                        if not data:selector.unregister(channel);channel.close();continue
                        if key.data=='error':
                            errors.extend(data)
                            if len(errors)>65536:raise EvidenceError('SSH diagnostics exceeded bound')
                            if fatal_diagnostic(errors):raise EvidenceError('SSH authentication or identity failure; diagnostics withheld')
                        else:
                            out_count+=len(data)
                            if out_count>maximum:raise EvidenceError('SSH transfer output exceeded bound')
                            destination.write(data)
                now=self.monotonic();counts=(in_count,out_count)
                if progress is not None and counts!=last_counts and now-last_observed>=2:
                    progress({'bytes_sent':in_count,'bytes_received':out_count});last_counts=counts;last_observed=now
            left=min(end-self.monotonic(),deadline-self.wall())
            if left<=0:raise deadline_failure(process,errors)
            try:code=process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                raise deadline_failure(process,errors) from None
            if code!=0:raise process_failure(code,errors)
            if not input_done:raise EvidenceError('SSH transfer input incomplete')
            if progress is not None and (in_count,out_count)!=last_counts:progress({'bytes_sent':in_count,'bytes_received':out_count})
            return {'bytes_sent':in_count,'bytes_received':out_count,'process_exit_code':0,'pod_id':self.profile['pod_id'],
                    'endpoint_observation_sha256':self.profile['endpoint_observation_sha256'],'host_key_trust':self.profile['host_key_trust']}
        finally:
            try:
                if selector is not None:selector.close()
            finally:
                try:
                    if process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        try:process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            process.wait(timeout=2)
                finally:
                    for channel in (process.stdin,process.stdout,process.stderr):
                        if channel is not None and not channel.closed:channel.close()

    def inspect(self,name,maximum,deadline):
        import io
        relative(name);integer(maximum,1,2**40,'remote inventory byte bound');reply=io.BytesIO()
        self.stream(['/usr/bin/python3','-c',REMOTE_GET,self.profile['remote_root'],name,str(maximum),'describe'],reply,4096,deadline)
        value=parse_json(reply.getvalue(),canonical_required=False)
        if value=={'path':name,'present':False}:return None
        fields(value,'path present bytes sha256','SSH peer inventory');require_digest(value['sha256'])
        integer(value['bytes'],0,maximum,'remote inventory file size')
        if value['path']!=name or value['present'] is not True:raise EvidenceError('SSH peer inventory differs')
        return {k:value[k] for k in ('path','bytes','sha256')}

    def read_live(self,name,maximum,deadline):
        """One open file description for atomic writer metadata; no trust credit.

        Inspect-then-fetch is for immutable selected files. Mutable status can
        be atomically replaced between those calls, so retain one bounded read.
        """
        import base64,io
        relative(name);integer(maximum,1,16*1024**2,'live metadata bound');reply=io.BytesIO()
        self.stream(['/usr/bin/python3','-c',REMOTE_GET,self.profile['remote_root'],name,str(maximum),'observe'],reply,4*((maximum+2)//3)+4096,deadline)
        value=parse_json(reply.getvalue(),canonical_required=False)
        if value=={'path':name,'present':False}:return None
        fields(value,'path present bytes sha256 data_b64','live SSH peer bytes');require_digest(value['sha256'])
        integer(value['bytes'],0,maximum,'live observation size')
        if value['path']!=name or value['present'] is not True or type(value['data_b64']) is not str:
            raise EvidenceError('live SSH peer selection differs')
        try:data=base64.b64decode(value['data_b64'],validate=True)
        except Exception:raise EvidenceError('invalid live SSH encoding') from None
        if len(data)!=value['bytes'] or hashlib.sha256(data).hexdigest()!=value['sha256']:
            raise EvidenceError('live SSH bytes differ from their framing')
        return data

    def get(self,name,destination,expected,deadline,*,progress=None):
        relative(name);fields(expected,'path bytes sha256','expected transfer');require_digest(expected['sha256'])
        integer(expected['bytes'],0,2**40,'expected transfer size')
        if expected['path']!=name:raise EvidenceError('transfer path differs from selected inventory')
        destination=Path(destination)
        if destination.exists() or destination.is_symlink():raise EvidenceError('download requires fresh destination')
        partial=destination.with_name(destination.name+'.partial')
        try:fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        except FileExistsError:raise EvidenceError('preserved partial from a prior attempt; select a fresh destination') from None
        with os.fdopen(fd,'wb') as f:
            result=self.stream(['/usr/bin/python3','-c',REMOTE_GET,self.profile['remote_root'],name,str(expected['bytes']),'get'],f,expected['bytes'],deadline,progress=progress)
            f.flush();os.fsync(f.fileno())
        if result['bytes_received']!=expected['bytes'] or file_hash(partial)!=expected['sha256']:
            raise EvidenceError('downloaded SSH bytes differ; partial preserved')
        # An exclusive link avoids replacing a competing caller's destination.
        os.link(partial,destination);partial.unlink()
        dfd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
        return {**result,'sha256':expected['sha256'],'bytes':expected['bytes'],'scope':'actual selected bytes transferred; no training verification'}

    def put(self,name,source,deadline,*,replace=False,progress=None):
        import io
        relative(name);source=Path(source)
        if source.is_symlink() or not source.is_file():raise EvidenceError('regular upload source required')
        if type(replace) is not bool or replace and name not in ('external-progress-policies.json','request-stop'):
            raise EvidenceError('only explicit acknowledgement/stop controls may be replaced')
        size=source.stat().st_size;integer(size,0,2**40,'upload size');root=file_hash(source);reply=io.BytesIO()
        argv=['/usr/bin/python3','-c',REMOTE_PUT,self.profile['remote_root'],name,str(size),root,'1' if replace else '0']
        with source.open('rb') as f:result=self.stream(argv,reply,4096,deadline,source=f,source_bytes=size,progress=progress)
        value=parse_json(reply.getvalue(),canonical_required=False)
        if value!={'bytes':size,'sha256':root} or result['bytes_sent']!=size or file_hash(source)!=root:
            raise EvidenceError('remote acknowledgement or local upload source changed')
        return {**result,'bytes':size,'sha256':root,'scope':'SSH peer acknowledged complete file bytes; recorder must separately verify public ancestry'}
