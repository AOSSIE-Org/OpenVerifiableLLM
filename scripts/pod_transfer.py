#!/usr/bin/env python3
"""Bounded SSH file transport for an already adopted pod; never provisioning.

The caller independently selects the pod/endpoint observation and host-key pin.
Host-key TOFU is not provider attestation. Actual byte transfers can supply cost
progress; process liveness and remote clocks cannot. Private keys are local only.
"""
from pathlib import Path,PurePosixPath
import hashlib
import ipaddress
import math
import os
import re
import selectors
import shlex
import signal
import stat
import subprocess
import time

from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,parse_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer


class TransientTransportError(EvidenceError):
    """Failed transport only; no identity, framing or content verification credit."""


class RangeRecoveryExhausted(EvidenceError):
    """The finite range retry budget cannot be reset by snapshot re-entry."""


class SmallReadRecoveryExhausted(EvidenceError):
    """A zero-payload read exhausted its allowance; snapshot re-entry is fatal."""


def retryable_transport(error):
    """A cleanup failure or missing actual counters cannot authorize a retry."""
    counts=getattr(error,'transfer_counts',None)
    return (type(error) is TransientTransportError and not hasattr(error,'transport_cleanup_diagnostic')
        and type(counts) is dict and set(counts)=={'bytes_sent','bytes_received'}
        and all(type(v) is int and 0<=v<=2**40 for v in counts.values()))


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
        rb'timeout, server [0-9.]+ not responding\.',
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
    if errors:
        if not errors.endswith(b'\n'):
            return EvidenceError('incomplete SSH diagnostic at deadline; diagnostics withheld')
        return process_failure(255,errors)
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
root,name,size,action,*extra=sys.argv[1:];size=int(size)
if action not in ('get','describe','observe','range'):raise ValueError('action')
if action=='range':
 if len(extra)!=2:raise ValueError('range arguments')
 offset,length=map(int,extra)
 if not 0<=offset<size or not 0<length<=64*1024**2 or offset+length>size:raise ValueError('range bounds')
elif extra:raise ValueError('unexpected arguments')
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
 if not stat.S_ISREG(info.st_mode) or not 0<=info.st_size<=size or action in ('get','range') and info.st_size!=size:raise ValueError('file length/type')
 if action=='range':
  f.seek(offset);count=0
  while count<length:
   data=f.read(min(1024*1024,length-count))
   if not data:raise ValueError('short range')
   count+=len(data);sys.stdout.buffer.write(data)
  after=os.fstat(f.fileno())
  if (after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(info.st_size,info.st_mtime_ns,info.st_ctime_ns):raise ValueError('file changed')
  sys.stdout.buffer.flush();raise SystemExit(0)
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

# Large immutable downloads rotate bounded read-only connections. These are
# operational limits, not new copy/phase/rental deadlines or trust evidence.
RANGE_BYTES=64*1024**2
SMALL_READ_BYTES=16*1024**2
RANGE_SECONDS=90
RANGE_RETRIES=2


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
        # OpenSSH logs server-alive expiry at INFO, so ERROR hides the reason
        # for exit255. Keep diagnostics in the bounded private buffer; never print
        # them or broaden the closed retry classification.
        return ['ssh','-F','/dev/null','-T','-o','LogLevel=INFO','-i',str(self.key.resolve()),'-p',str(self.profile['port']),
            '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','ForwardAgent=no',
            '-o','ClearAllForwardings=yes','-o','ControlMaster=no','-o','ControlPath=none',
            '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(self.known.resolve()),
            '-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
            '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','root@'+self.profile['host'],
            'exec '+shlex.join(remote_argv)]

    def stream(self,argv,destination,maximum,deadline,*,source=None,source_bytes=None,progress=None,payload_idle_seconds=None,monotonic_deadline=None):
        integer(maximum,0,2**40,'transfer output bound');integer(deadline,1,2**53-1,'transfer deadline')
        if payload_idle_seconds is not None:
            integer(payload_idle_seconds,1,3600,'payload inactivity allowance')
            if source is not None:raise EvidenceError('payload inactivity applies only to read-only transfers')
        if source is not None:integer(source_bytes,0,2**40,'transfer input bound')
        remaining=deadline-self.wall()
        if remaining<=0:raise EvidenceError('transfer deadline expired')
        end=self.monotonic()+remaining
        if monotonic_deadline is not None:
            if type(monotonic_deadline) not in (int,float) or not math.isfinite(monotonic_deadline):
                raise EvidenceError('finite original monotonic deadline required')
            end=min(end,monotonic_deadline)
            if self.monotonic()>=end:raise EvidenceError('original monotonic transfer deadline expired')
        out_count=in_count=0;errors=bytearray();pending=b'';input_done=source is None
        last_payload=self.monotonic()
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
                if payload_idle_seconds is not None:left=min(left,last_payload+payload_idle_seconds-self.monotonic())
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
                        if not data:
                            if key.data=='error' and fatal_diagnostic(errors+b'\n'):
                                raise EvidenceError('SSH authentication or identity failure; diagnostics withheld')
                            selector.unregister(channel);channel.close();continue
                        if key.data=='error':
                            errors.extend(data)
                            if len(errors)>65536:raise EvidenceError('SSH diagnostics exceeded bound')
                            if fatal_diagnostic(errors):raise EvidenceError('SSH authentication or identity failure; diagnostics withheld')
                        else:
                            out_count+=len(data)
                            if out_count>maximum:
                                error=EvidenceError('SSH transfer output exceeded bound')
                                error.transfer_overflow=data
                                raise error
                            # Only actual bounded payload renews this local idle
                            # allowance. The parent deadline never moves; stderr,
                            # callbacks and duplicate health credit cannot renew it.
                            last_payload=self.monotonic()
                            destination.write(data)
                now=self.monotonic();counts=(in_count,out_count)
                if progress is not None and counts!=last_counts and now-last_observed>=2:
                    progress({'bytes_sent':in_count,'bytes_received':out_count});last_counts=counts;last_observed=now
            left=min(end-self.monotonic(),deadline-self.wall())
            if payload_idle_seconds is not None:left=min(left,last_payload+payload_idle_seconds-self.monotonic())
            if left<=0:raise deadline_failure(process,errors)
            try:code=process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                raise deadline_failure(process,errors) from None
            if code!=0:raise process_failure(code,errors)
            if not input_done:raise EvidenceError('SSH transfer input incomplete')
            if progress is not None and (in_count,out_count)!=last_counts:progress({'bytes_sent':in_count,'bytes_received':out_count})
            if min(end-self.monotonic(),deadline-self.wall())<=0:raise deadline_failure(process,errors)
            return {'bytes_sent':in_count,'bytes_received':out_count,'process_exit_code':0,'pod_id':self.profile['pod_id'],
                    'endpoint_observation_sha256':self.profile['endpoint_observation_sha256'],'host_key_trust':self.profile['host_key_trust']}
        except Exception as error:
            error.transfer_counts={'bytes_sent':in_count,'bytes_received':out_count}
            try:
                from private_transport_diagnostics import bounded_bytes
                error.transport_diagnostic={'stderr':bounded_bytes(errors),'process_exit_code':process.poll(),
                    'bytes_sent':in_count,'bytes_received':out_count,'deadline_epoch':deadline}
            except Exception:pass  # diagnostics cannot replace the primary error
            raise
        finally:
            import sys
            primary=sys.exc_info()[1]
            try:
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
                        channel_error=None
                        for channel in (process.stdin,process.stdout,process.stderr):
                            try:
                                if channel is not None and not channel.closed:channel.close()
                            except Exception as error:
                                if channel_error is None:channel_error=error
                        if channel_error is not None:raise channel_error
            except Exception as cleanup:
                if primary is None:raise
                primary.transport_cleanup_diagnostic={'exception_class':type(cleanup).__name__,'message':str(cleanup)[:4096]}

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
        try:return self._get(name,destination,expected,deadline,progress=progress)
        except Exception as error:
            partial=Path(destination).with_name(Path(destination).name+'.partial')
            # A fresh snapshot must not discard an observed immutable prefix.
            # Range retries compare prefixes within the same owned operation.
            error.immutable_download_bytes=partial.stat().st_size if partial.is_file() and not partial.is_symlink() else None
            raise

    def _get(self,name,destination,expected,deadline,*,progress=None):
        relative(name);fields(expected,'path bytes sha256','expected transfer');require_digest(expected['sha256'])
        integer(expected['bytes'],0,2**40,'expected transfer size')
        integer(deadline,1,2**53-1,'transfer deadline')
        end=self.monotonic()+max(0,deadline-self.wall())
        if expected['path']!=name:raise EvidenceError('transfer path differs from selected inventory')
        validate(self.profile,self.key,self.known)
        profile_root=digest(self.profile);selected=dict(expected)
        selected_key=self.key.resolve();selected_known=self.known.resolve();key_root=file_hash(self.key)
        def check_selection():
            if (digest(self.profile)!=profile_root or expected!=selected
                or self.key.resolve()!=selected_key or self.known.resolve()!=selected_known
                or file_hash(self.key)!=key_root):raise EvidenceError('immutable download selection changed')
            validate(self.profile,self.key,self.known)
        destination=Path(destination)
        if destination.exists() or destination.is_symlink():raise EvidenceError('download requires fresh destination')
        partial=destination.with_name(destination.name+'.partial')
        try:fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        except FileExistsError:raise EvidenceError('preserved partial from a prior attempt; select a fresh destination') from None
        with os.fdopen(fd,'wb') as f:
            if expected['bytes']>min(SMALL_READ_BYTES,RANGE_BYTES):
                result=self._get_ranges(name,f,partial,expected,deadline,end,progress,check_selection)
            else:
                result=self._get_small(name,f,partial,expected,deadline,end,progress,check_selection)
            f.flush();os.fsync(f.fileno())
        check_selection()
        if result['bytes_received']!=expected['bytes'] or file_hash(partial)!=expected['sha256']:
            raise EvidenceError('downloaded SSH bytes differ; partial preserved')
        if self.wall()>=deadline or self.monotonic()>=end:raise EvidenceError('download verification exceeded original deadline')
        # An exclusive link avoids replacing a competing caller's destination.
        os.link(partial,destination);partial.unlink()
        dfd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
        if self.wall()>=deadline or self.monotonic()>=end:raise EvidenceError('download installation exceeded original deadline')
        return {**result,'sha256':expected['sha256'],'bytes':expected['bytes'],'scope':'actual selected bytes transferred; no training verification'}

    def _get_small(self,name,destination,partial,expected,deadline,end,progress,check_selection):
        # Retrying only a positively classified zero-byte read cannot discard a
        # selected immutable prefix. Writes, unknown errors and partial reads
        # still fail immediately. The caller's original deadlines are binding.
        failures=[]
        while True:
            check_selection()
            left=min(deadline-self.wall(),end-self.monotonic())
            if left<=0:raise EvidenceError('small read exhausted original transfer deadline')
            attempt_deadline=min(deadline,int(self.wall()+left))
            if attempt_deadline<=self.wall():raise EvidenceError('insufficient time for another bounded read')
            try:
                result=self.stream(['/usr/bin/python3','-c',REMOTE_GET,self.profile['remote_root'],name,str(expected['bytes']),'get'],destination,expected['bytes'],attempt_deadline,progress=progress,monotonic_deadline=end)
            except Exception as error:
                if not (retryable_transport(error) and error.transfer_counts=={'bytes_sent':0,'bytes_received':0}
                        and destination.tell()==0 and os.fstat(destination.fileno()).st_size==0
                        and not getattr(error,'transfer_overflow',b'')):
                    raise
                check_selection()
                failures.append({'attempt':len(failures),'deadline_epoch':attempt_deadline,
                    'error_type':type(error).__name__,'bytes_received':0,'bytes_sent':0})
                saved=partial.with_name(partial.name+'.read-attempts')
                saved.mkdir(mode=0o700,exist_ok=True)
                write_json(saved/f'failure-{len(failures):03d}.json',{**failures[-1],
                    'transport_diagnostic':getattr(error,'transport_diagnostic',None)})
                if len(failures)>2:
                    raise SmallReadRecoveryExhausted('immutable small read retry budget exhausted; preserve attempts') from error
                delay=2**len(failures)
                if min(deadline-self.wall(),end-self.monotonic())<=delay:
                    raise SmallReadRecoveryExhausted('original deadline cannot fit remaining read recovery; preserve attempts') from error
                time.sleep(delay)
                continue
            return {**result,'zero_payload_read_failures':failures,
                    'small_read_policy':{'maximum_retries':2,'original_deadline_epoch':deadline}}

    def _get_ranges(self,name,destination,partial,expected,deadline,end,progress,check_selection):
        """Retry only failed read-only ranges, never a start/write or old partial.

        Successful ranges are buffered within64MiB then appended once. A failed
        range's bytes and receipt remain private beside the preserved aggregate
        partial. Only final complete size/hash verification can install a file.
        Progress uses logical high-water bytes, so retransmissions earn no extra
        health credit; actual payload bytes remain separately accounted in receipts.
        """
        import io
        offset=payload=failures=0;attempts=[];last=None;observed_prefix=b''
        directory=partial.with_name(partial.name+'.ranges')
        directory.mkdir(mode=0o700)
        while offset<expected['bytes']:
            check_selection()
            left=min(deadline-self.wall(),end-self.monotonic())
            if left<=0:raise EvidenceError('range recovery exhausted original transfer deadline')
            length=min(RANGE_BYTES,expected['bytes']-offset)
            attempt_deadline=min(deadline,int(self.wall()+left))
            if attempt_deadline<=self.wall():raise EvidenceError('insufficient time for another bounded range')
            data=io.BytesIO();index=len(attempts);completed=False
            def report(counts):
                if progress is not None:progress({'bytes_sent':0,'bytes_received':offset+counts['bytes_received']})
            try:
                last=self.stream(['/usr/bin/python3','-c',REMOTE_GET,self.profile['remote_root'],name,str(expected['bytes']),'range',str(offset),str(length)],data,length,attempt_deadline,progress=report,payload_idle_seconds=RANGE_SECONDS,monotonic_deadline=end)
                completed=True
                check_selection()
                if last['bytes_received']!=length or len(data.getbuffer())!=length:
                    raise EvidenceError('downloaded SSH range length differs')
                if data.getvalue()[:len(observed_prefix)]!=observed_prefix:
                    raise EvidenceError('immutable range returned conflicting bytes')
            except Exception as error:
                saved_bytes=data.getvalue()+getattr(error,'transfer_overflow',b'')
                counts=getattr(error,'transfer_counts',None)
                if counts is None and completed:counts={k:last[k] for k in ('bytes_sent','bytes_received')}
                received=counts['bytes_received'] if type(counts) is dict and type(counts.get('bytes_received')) is int else None
                if received is not None:payload+=received
                saved=directory/f'attempt-{index:05d}.partial'
                with saved.open('xb') as f:f.write(saved_bytes);f.flush();os.fsync(f.fileno())
                saved.chmod(0o600)
                receipt={'offset':offset,'bytes_requested':length,'bytes_received':received,'saved_bytes':len(saved_bytes),'deadline_epoch':attempt_deadline,'result':'FAILED','error_type':type(error).__name__,'partial_sha256':file_hash(saved)}
                write_json(directory/f'attempt-{index:05d}.json',receipt);attempts.append(receipt)
                overlap=min(len(observed_prefix),len(saved_bytes))
                if observed_prefix[:overlap]!=saved_bytes[:overlap]:
                    raise EvidenceError('immutable range returned conflicting bytes') from error
                if not retryable_transport(error):raise
                if received!=len(saved_bytes):raise EvidenceError('incomplete range payload accounting') from error
                if len(saved_bytes)>len(observed_prefix):observed_prefix=saved_bytes
                failures+=1
                if failures>RANGE_RETRIES:raise RangeRecoveryExhausted('immutable range retry budget exhausted; preserve attempts') from error
                if min(deadline-self.wall(),end-self.monotonic())<=0:raise
                continue
            payload+=length;destination.write(data.getbuffer());destination.flush()
            receipt={'offset':offset,'bytes_requested':length,'bytes_received':length,'deadline_epoch':attempt_deadline,'result':'TRANSFERRED_NOT_YET_WHOLE_FILE_VERIFIED'}
            write_json(directory/f'attempt-{index:05d}.json',receipt);attempts.append(receipt)
            offset+=length;observed_prefix=b''
        return {**last,'bytes_received':offset,'transferred_payload_bytes':payload,'range_attempts':attempts,
                'range_policy':{'bytes':RANGE_BYTES,'payload_idle_seconds':RANGE_SECONDS,'maximum_retries':RANGE_RETRIES,'original_deadline_epoch':deadline}}

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
