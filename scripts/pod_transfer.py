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
import hashlib,json,os,stat,sys
from pathlib import Path
root,name,size,action=sys.argv[1:];size=int(size)
if action not in ('get','describe'):raise ValueError('action')
if not root.startswith('/') or '..' in Path(root).parts or str(Path(root))!=root or any(x in ('','.','..') for x in name.split('/')):raise ValueError('path')
path=Path(root)/name;current=Path('/')
for part in path.parts[1:]:
 current=current/part
 if current.is_symlink():raise ValueError('symlink path')
if action=='describe' and not path.exists():
 print(json.dumps({'path':name,'present':False},sort_keys=True,separators=(',',':')));raise SystemExit(0)
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
with os.fdopen(fd,'rb') as f:
 info=os.fstat(f.fileno())
 if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=size or action=='get' and info.st_size!=size:raise ValueError('file length/type')
 count=0;h=hashlib.sha256()
 while True:
  data=f.read(min(1024*1024,size-count+1))
  if not data:break
  count+=len(data)
  if count>size:raise ValueError('file grew')
  h.update(data)
  if action=='get':sys.stdout.buffer.write(data)
 if count!=info.st_size:raise ValueError('file length changed')
 if action=='describe':print(json.dumps({'path':name,'present':True,'bytes':count,'sha256':h.hexdigest()},sort_keys=True,separators=(',',':')))
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
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*',name) or any(x in ('.','..') for x in name.split('/')):
        raise EvidenceError('unsafe remote relative path')
    if str(PurePosixPath(name))!=name:raise EvidenceError('noncanonical remote path')
    return name


class Transport:
    def __init__(self,profile,key,known_hosts,*,popen=subprocess.Popen,wall=time.time,monotonic=time.monotonic):
        self.profile=validate(profile,key,known_hosts);self.key=Path(key);self.known=Path(known_hosts)
        self.popen=popen;self.wall=wall;self.monotonic=monotonic

    def command(self,remote_argv):
        validate(self.profile,self.key,self.known)
        return ['ssh','-F','/dev/null','-T','-q','-i',str(self.key.resolve()),'-p',str(self.profile['port']),
            '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','ForwardAgent=no',
            '-o','ClearAllForwardings=yes','-o','ControlMaster=no','-o','ControlPath=none',
            '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(self.known.resolve()),
            '-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
            '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','root@'+self.profile['host'],
            'exec '+shlex.join(remote_argv)]

    def stream(self,argv,destination,maximum,deadline,*,source=None,source_bytes=None,progress=None):
        integer(maximum,0,2**40,'transfer output bound');integer(deadline,1,2**53-1,'transfer deadline')
        if source is not None:integer(source_bytes,1,2**40,'transfer input bound')
        remaining=deadline-self.wall()
        if remaining<=0:raise EvidenceError('transfer deadline expired')
        end=self.monotonic()+remaining;out_count=in_count=0;errors=bytearray();pending=b'';input_done=source is None
        process=self.popen(self.command(argv),stdin=subprocess.DEVNULL if source is None else subprocess.PIPE,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,start_new_session=True)
        selector=selectors.DefaultSelector()
        for stream,kind in [(process.stdout,'output'),(process.stderr,'error')]:
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,kind)
        if source is not None:
            os.set_blocking(process.stdin.fileno(),False);selector.register(process.stdin,selectors.EVENT_WRITE,'input')
        last_observed=self.monotonic();last_counts=(0,0)
        try:
            while selector.get_map():
                left=min(end-self.monotonic(),deadline-self.wall())
                if left<=0:raise EvidenceError('transfer deadline reached; rental deadline is unchanged')
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
                        else:
                            out_count+=len(data)
                            if out_count>maximum:raise EvidenceError('SSH transfer output exceeded bound')
                            destination.write(data)
                now=self.monotonic();counts=(in_count,out_count)
                if progress is not None and counts!=last_counts and now-last_observed>=2:
                    progress({'bytes_sent':in_count,'bytes_received':out_count});last_counts=counts;last_observed=now
            left=min(end-self.monotonic(),deadline-self.wall())
            if left<=0:raise EvidenceError('transfer deadline reached before process exit')
            if process.wait(timeout=left)!=0 or not input_done:raise EvidenceError('SSH transfer process failed; diagnostics withheld')
            if progress is not None and (in_count,out_count)!=last_counts:progress({'bytes_sent':in_count,'bytes_received':out_count})
            return {'bytes_sent':in_count,'bytes_received':out_count,'process_exit_code':0,'pod_id':self.profile['pod_id'],
                    'endpoint_observation_sha256':self.profile['endpoint_observation_sha256'],'host_key_trust':self.profile['host_key_trust']}
        finally:
            selector.close()
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait(timeout=2)
            for channel in (process.stdin,process.stdout,process.stderr):
                if channel is not None and not channel.closed:channel.close()

    def inspect(self,name,maximum,deadline):
        import io
        relative(name);integer(maximum,1,2**40,'remote inventory byte bound');reply=io.BytesIO()
        self.stream(['/usr/bin/python3','-c',REMOTE_GET,self.profile['remote_root'],name,str(maximum),'describe'],reply,4096,deadline)
        value=parse_json(reply.getvalue(),canonical_required=False)
        if value=={'path':name,'present':False}:return None
        fields(value,'path present bytes sha256','SSH peer inventory');require_digest(value['sha256'])
        integer(value['bytes'],1,maximum,'remote inventory file size')
        if value['path']!=name or value['present'] is not True:raise EvidenceError('SSH peer inventory differs')
        return {k:value[k] for k in ('path','bytes','sha256')}

    def get(self,name,destination,expected,deadline,*,progress=None):
        relative(name);fields(expected,'path bytes sha256','expected transfer');require_digest(expected['sha256'])
        integer(expected['bytes'],1,2**40,'expected transfer size')
        if expected['path']!=name:raise EvidenceError('transfer path differs from selected inventory')
        destination=Path(destination)
        if destination.exists() or destination.is_symlink():raise EvidenceError('download requires fresh destination')
        partial=destination.with_name(destination.name+'.partial')
        fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
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
        size=source.stat().st_size;integer(size,1,2**40,'upload size');root=file_hash(source);reply=io.BytesIO()
        argv=['/usr/bin/python3','-c',REMOTE_PUT,self.profile['remote_root'],name,str(size),root,'1' if replace else '0']
        with source.open('rb') as f:result=self.stream(argv,reply,4096,deadline,source=f,source_bytes=size,progress=progress)
        value=parse_json(reply.getvalue(),canonical_required=False)
        if value!={'bytes':size,'sha256':root} or result['bytes_sent']!=size or file_hash(source)!=root:
            raise EvidenceError('remote acknowledgement or local upload source changed')
        return {**result,'bytes':size,'sha256':root,'scope':'SSH peer acknowledged complete file bytes; recorder must separately verify public ancestry'}
