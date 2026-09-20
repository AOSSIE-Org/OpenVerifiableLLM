"""Bounded off-pod job launch/adoption, observation and exact export transport.

The operator selects the job and worker digests independently. An uncertain start
is only reconciled read-only; it cannot automatically start another process.
No RunPod/HF credentials or private signing seed are transferred by this client.
"""
from pathlib import Path
import io
import os
import uuid

from ovl_pipeline.canonical import EvidenceError,atomic_write,confined,digest,file_hash,parse_json,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer
from pod_transfer import relative
from pod_observe import observe_many


# Prospective control-operation allowance, always clipped by the caller to the
# existing job/rental deadline. It does not grant progress or permit a new start.
LAUNCH_CONTROL_SECONDS = 120


REMOTE_TREE=r'''
import hashlib,json,os,stat,sys
from pathlib import Path
root,name,missing,selection=sys.argv[1:];base=Path(root)/name
if selection not in ('root','subtree') or (selection=='root' and name!='.'):raise ValueError('root selection')
if not Path(root).is_absolute() or '..' in Path(root).parts or (selection!='root' and any(x in ('','.','..') for x in name.split('/'))):raise ValueError('path')
for p in [base,*base.parents]:
 if p.is_symlink():raise ValueError('symlink root')
if not base.exists() and missing=='allow':sys.stdout.write('[]');raise SystemExit(0)
if not base.is_dir():raise ValueError('directory')
files=[];total=0
for parent,dirs,names in os.walk(base,followlinks=False):
 dirs.sort();names.sort()
 for n in dirs:
  if (Path(parent)/n).is_symlink():raise ValueError('symlink directory')
 for n in names:
  p=Path(parent)/n
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
  with os.fdopen(fd,'rb') as f:
   s=os.fstat(f.fileno())
   if not stat.S_ISREG(s.st_mode):raise ValueError('regular file')
   h=hashlib.sha256();count=0
   while True:
    b=f.read(1024*1024)
    if not b:break
    count+=len(b);total+=len(b)
    if count>s.st_size or total>2**40:raise ValueError('changing or oversized tree')
    h.update(b)
   if count!=s.st_size:raise ValueError('changed file length')
  files.append({'path':p.relative_to(base).as_posix(),'bytes':count,'sha256':h.hexdigest()})
  if len(files)>100000:raise ValueError('too many files')
files.sort(key=lambda f:f['path'])
value=json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
if len(value)>16*1024**2:raise ValueError('inventory too large')
sys.stdout.buffer.write(value)
'''


def save_once(path,value):
    if path.exists():
        if read_json(path)!=value:raise EvidenceError('retained job selection changed')
    else:write_json(path,value)


def launch(transport,job_file,expected_job,worker_file,expected_worker,output,deadline,*,before_start=None):
    require_digest(expected_job);require_digest(expected_worker)
    job_file=Path(job_file);worker_file=Path(worker_file);output=Path(output)
    if job_file.is_symlink() or worker_file.is_symlink() or file_hash(job_file)!=expected_job or file_hash(worker_file)!=expected_worker:
        raise EvidenceError('job or runner differs from independent operator selection')
    # The worker validates this selected descriptor before spawning. The caller
    # must separately enforce production admission before authorizing its hash.
    job=read_json(job_file)
    if digest(job)!=expected_job:raise EvidenceError('canonical operator-selected job required')
    output.mkdir(mode=0o700,parents=True,exist_ok=True)
    intent={'schema':'ovl.offpod-job-launch-intent.v1','job_sha256':expected_job,'worker_sha256':expected_worker,
            'profile_sha256':digest(transport.profile)}
    journal=output/'launch-intent.json';existing=journal.exists()
    if existing:
        save_once(journal,intent)
        return reconcile_launch(transport,intent,output,deadline)
    remote_job='jobs/'+expected_job;remote_worker='tools/pod-job-worker-'+expected_worker+'.py'
    transport.put(remote_worker,worker_file,deadline)
    transport.put(remote_job+'/job.json',job_file,deadline)
    if before_start is not None:before_start()
    # No subsequent restart is allowed to repeat start, including a crash before
    # sending it. Transfers of immutable inputs before this fence are harmless.
    save_once(journal,intent)
    if before_start is not None:before_start()
    reply=io.BytesIO()
    outcome={'completed':False}
    try:
        transport.stream(['/usr/bin/python3',transport.profile['remote_root']+'/'+remote_worker,'start',
                          transport.profile['remote_root']+'/'+remote_job,expected_job,expected_worker],reply,65536,deadline)
        outcome={'completed':True}
    except BaseException as error:
        outcome['exception_type']=type(error).__name__
        raise
    finally:
        # Transport may deliver stdout and then fail. Keep received bytes even
        # then; a later good remote receipt cannot erase a received contradiction.
        atomic_write(output/'start-response.raw',reply.getvalue())
        save_once(output/'start-transport.json',outcome)
    retained_start_response(output,intent)
    return reconcile_launch(transport,intent,output,deadline)


def process_receipt(value):
    fields(value,'pid start_ticks process_group','remote process identity')
    for name in value:integer(value[name],1,2**53-1,'remote process '+name)
    if value['pid']!=value['process_group']:raise EvidenceError('remote process is not selected group leader')


def launch_receipt(value,intent):
    fields(value,'schema job_sha256 worker_sha256 runner','remote launch receipt')
    if value['schema']!='ovl.pod-job-launch.v1' or value['job_sha256']!=intent['job_sha256'] or value['worker_sha256']!=intent['worker_sha256']:
        raise EvidenceError('remote launch identity differs')
    process_receipt(value['runner'])


def retained_start_response(output,intent):
    output=Path(output);raw=output/'start-response.raw';parsed=output/'start-response.json'
    outcome=output/'start-transport.json'
    if outcome.exists() and read_json(outcome)['completed'] and (not raw.exists() or not raw.stat().st_size):
        raise EvidenceError('completed launch transport lacks acknowledgement')
    if raw.exists() and raw.stat().st_size:
        # Nonempty malformed/partial replies remain unresolved, never silently
        # discarded as a missing acknowledgement during read-only adoption.
        if raw.stat().st_size>65536:raise EvidenceError('retained launch acknowledgement exceeds bound')
        response=parse_json(raw.read_bytes(),canonical_required=False)
        save_once(parsed,response)
    if parsed.exists():
        response=read_json(parsed);launch_receipt(response,intent);return response
    return None


def reconcile_launch(transport,intent,output,deadline):
    if intent['profile_sha256']!=digest(transport.profile):raise EvidenceError('job endpoint identity changed')
    response=retained_start_response(output,intent)
    observation=Path(output)/('reconcile-'+uuid.uuid4().hex);observation.mkdir(mode=0o700)
    job='jobs/'+intent['job_sha256']
    # One read-only exchange retains all three individually checked receipts.
    # Preserve the caller's original deadline and every identity check; batching
    # is neither an atomic snapshot nor authority to resend a fenced start.
    names=('intent.json','receipt.json','child.json')
    bundle=observe_many(transport,{job+'/launch/'+name:observation/name for name in names},65536,deadline)
    selected=bundle[job+'/launch/intent.json']
    expected={'schema':'ovl.pod-job-launch-intent.v1','job_sha256':intent['job_sha256'],'worker_sha256':intent['worker_sha256']}
    if selected!=expected:raise EvidenceError('uncertain launch is not yet reconcilable; never automatically restart')
    value=bundle[job+'/launch/receipt.json']
    child=bundle[job+'/launch/child.json']
    if value is not None:
        launch_receipt(value,intent)
        if response is not None and response!=value:raise EvidenceError('start acknowledgement contradicts retained launch receipt')
    if child is not None:
        fields(child,'schema job_sha256 process','remote child receipt')
        if child['schema']!='ovl.pod-job-child.v1' or child['job_sha256']!=intent['job_sha256']:raise EvidenceError('remote child identity differs')
        process_receipt(child['process'])
    if value is None and child is None:raise EvidenceError('launch receipt absent; preserve and observe, no new start')
    state=job_supervision(transport,intent['job_sha256'],intent['worker_sha256'],deadline)
    write_json(observation/'supervision.json',state)
    result={'schema':'ovl.offpod-job-adoption.v1','job_sha256':intent['job_sha256'],'launch':value,'child':child,
            'state':state['state'],'supervision':state,
            'scope':'retained job identity and bounded process observation; no runtime or training verification','reissued_start':False}
    write_json(observation/'adoption.json',result);return result


def job_supervision(transport,job,worker,deadline,*,abandon=False):
    require_digest(job);require_digest(worker);reply=io.BytesIO();base=transport.profile['remote_root']
    transport.stream(['/usr/bin/python3',base+'/tools/pod-job-worker-'+worker+'.py',
                      'abandon' if abandon else 'inspect',base+'/jobs/'+job,job,worker],reply,65536,deadline)
    value=parse_json(reply.getvalue(),canonical_required=False)
    if abandon:
        if value.get('schema') not in ('ovl.workload-job-abandonment.v1','ovl.workload-job-exit.v1') or value.get('job_sha256')!=job:
            raise EvidenceError('wrong abandonment response')
    else:
        fields(value,'schema job_sha256 state runner_alive child_alive terminal scope','job supervision response')
        states=('RUNNING_UNVERIFIED','STARTING_UNVERIFIED','EXITED','ABANDONED','CHILD_IDENTITY_UNKNOWN',
                'LAUNCH_FENCE_WITHOUT_INTENT','LAUNCH_NOT_OBSERVED','SUPERVISOR_ABSENT')
        if value['schema']!='ovl.pod-job-supervision.v1' or value['job_sha256']!=job or value['state'] not in states:
            raise EvidenceError('wrong job supervision identity/state')
        if type(value['runner_alive']) is not bool or type(value['child_alive']) is not bool:
            raise EvidenceError('invalid job liveness response')
    return value


def tree(transport,name,deadline,*,allow_missing=False,whole_root=False):
    if whole_root:
        if name!='.':raise EvidenceError('whole profile root requires explicit dot selection')
    else:relative(name)
    buffer=io.BytesIO()
    transport.stream(['/usr/bin/python3','-c',REMOTE_TREE,transport.profile['remote_root'],name,'allow' if allow_missing else 'reject','root' if whole_root else 'subtree'],buffer,16*1024**2,deadline)
    value=parse_json(buffer.getvalue(),canonical_required=False)
    if type(value) is not list or not value and not allow_missing or len(value)>100000:raise EvidenceError('nonempty bounded export inventory required')
    total=0;names=[]
    for f in value:
        fields(f,'path bytes sha256','export tree file');relative(f['path']);require_digest(f['sha256'])
        integer(f['bytes'],0,2**40,'export file bytes');total+=f['bytes'];names.append(f['path'])
    if names!=sorted(set(names)) or total>2**40:raise EvidenceError('duplicate/unsorted/oversized export tree')
    return value


def export_tree(transport,name,output,deadline,*,progress=None):
    """Preserve all peer-enumerated files; rehash again to reject changing trees.

    Export roots must be caller-selected from its job plan, never discovered from
    an untrusted result. Semantic checkpoint verification remains a separate gate.
    """
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    files=tree(transport,name,deadline);write_json(output/'inventory.json',files)
    target=output/'files';target.mkdir(mode=0o700)
    if len(files)>=8:
        from pod_bulk_export import receive
        batch=receive(transport,name,files,output/'bulk',deadline,progress=progress)
        for item in files:
            destination=confined(target,item['path']);destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            os.link(confined(Path(batch['files_directory']),item['path']),destination,follow_symlinks=False)
    else:
        for item in files:
            destination=confined(target,item['path']);destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            transport.get(name+'/'+item['path'],destination,{**item,'path':name+'/'+item['path']},deadline,
                          progress=None if progress is None else lambda counts,item=item:progress(digest({'tree':name,'file':item}),counts,item['bytes']))
    verify_inventory(target,files)
    after=tree(transport,name,deadline);write_json(output/'after-inventory.json',after)
    if after!=files:raise EvidenceError('remote export changed; preserve copies without completion')
    result={'schema':'ovl.offpod-tree-export.v1','result':'PASS','pod_id':transport.profile['pod_id'],
            'root':name,'files':files,'scope':'actual complete peer-enumerated bytes copied and rehashed; no semantic training/replay proof'}
    write_json(output/'export.json',result);return result
