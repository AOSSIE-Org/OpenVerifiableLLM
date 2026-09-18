"""Bounded off-pod job launch/adoption, observation and exact export transport.

The operator selects the job and worker digests independently. An uncertain start
is only reconciled read-only; it cannot automatically start another process.
No RunPod/HF credentials or private signing seed are transferred by this client.
"""
from pathlib import Path
import io
import uuid

from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,parse_json,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer
from pod_transfer import relative
from pod_checkpoint_handoff import observe


REMOTE_TREE=r'''
import hashlib,json,os,stat,sys
from pathlib import Path
root,name,missing=sys.argv[1:];base=Path(root)/name
if not Path(root).is_absolute() or '..' in Path(root).parts or any(x in ('','.','..') for x in name.split('/')):raise ValueError('path')
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


def launch(transport,job_file,expected_job,worker_file,expected_worker,output,deadline):
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
    # No subsequent restart is allowed to repeat start, including a crash before
    # sending it. Transfers of immutable inputs before this fence are harmless.
    save_once(journal,intent)
    reply=io.BytesIO()
    transport.stream(['/usr/bin/python3',transport.profile['remote_root']+'/'+remote_worker,'start',
                      transport.profile['remote_root']+'/'+remote_job,expected_job,expected_worker],reply,65536,deadline)
    response=parse_json(reply.getvalue(),canonical_required=False)
    save_once(output/'start-response.json',response)
    return reconcile_launch(transport,intent,output,deadline)


def reconcile_launch(transport,intent,output,deadline):
    if intent['profile_sha256']!=digest(transport.profile):raise EvidenceError('job endpoint identity changed')
    observation=Path(output)/('reconcile-'+uuid.uuid4().hex);observation.mkdir(mode=0o700)
    job='jobs/'+intent['job_sha256']
    selected=observe(transport,job+'/launch/intent.json',observation/'intent.json',65536,deadline,optional=True)
    expected={'schema':'ovl.pod-job-launch-intent.v1','job_sha256':intent['job_sha256'],'worker_sha256':intent['worker_sha256']}
    if selected!=expected:raise EvidenceError('uncertain launch is not yet reconcilable; never automatically restart')
    value=observe(transport,job+'/launch/receipt.json',observation/'receipt.json',65536,deadline,optional=True)
    child=observe(transport,job+'/launch/child.json',observation/'child.json',65536,deadline,optional=True)
    if value is not None:
        fields(value,'schema job_sha256 worker_sha256 runner','remote launch receipt')
        if value['schema']!='ovl.pod-job-launch.v1' or value['job_sha256']!=intent['job_sha256'] or value['worker_sha256']!=intent['worker_sha256']:
            raise EvidenceError('remote launch identity differs')
    if child is not None:
        fields(child,'schema job_sha256 process','remote child receipt')
        if child['schema']!='ovl.pod-job-child.v1' or child['job_sha256']!=intent['job_sha256']:raise EvidenceError('remote child identity differs')
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


def tree(transport,name,deadline,*,allow_missing=False):
    relative(name);buffer=io.BytesIO()
    transport.stream(['/usr/bin/python3','-c',REMOTE_TREE,transport.profile['remote_root'],name,'allow' if allow_missing else 'reject'],buffer,16*1024**2,deadline)
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
