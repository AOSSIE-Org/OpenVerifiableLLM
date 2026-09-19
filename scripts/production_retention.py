"""Preserve every declared production output after an observed terminal process.

No logs-only shortcut: the independently selected descriptor supplies all roots,
with separate pinned transports for sibling control and fresh record directories.
This establishes byte retention after observed exit, never training acceptance.
"""
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields
from pod_checkpoint_handoff import observe
from pod_job_client import job_supervision
from pod_versioned_export import export,regular_directory
from workload_health import terminal_status


def roots(control,transports,job,job_sha256):
    profiles=[control,*transports];selected={}
    identity=('pod_id','host','port','user','endpoint_observation_sha256','known_hosts_sha256','host_key_trust')
    for t in profiles:
        if any(t.profile[k]!=control.profile[k] for k in identity):raise EvidenceError('retention profiles refer to different selected endpoints')
        root=t.profile['remote_root']
        if root in selected:raise EvidenceError('duplicate selected retention profile')
        selected[root]=t
    declarations=[control.profile['remote_root']+'/jobs/'+job_sha256,*job['export_roots']]
    if len(set(declarations))!=len(declarations) or any(a!=b and b.startswith(a+'/') for a in declarations for b in declarations):
        raise EvidenceError('duplicate or overlapping declared retention roots')
    result=[]
    for root in declarations:
        matches=[p for p in selected if root==p or root.startswith(p+'/')]
        if len(matches)!=1:raise EvidenceError('declared output lacks one independently selected profile')
        profile=matches[0];name='.' if root==profile else root[len(profile)+1:]
        result.append((selected[profile],name,root))
    return result


def retain(control,transports,job_file,expected_job,expected_worker,store,output,deadline,*,progress=None):
    require_digest(expected_job);require_digest(expected_worker);job=read_json(Path(job_file))
    if digest(job)!=expected_job or job.get('kind') not in ('production-record','full-replay'):
        raise EvidenceError('selected production job descriptor required')
    selected=roots(control,transports,job,expected_job)
    output=regular_directory(output,fresh=True)
    supervision=job_supervision(control,expected_job,expected_worker,deadline)
    write_json(output/'supervision.json',supervision)
    if supervision['state'] not in ('EXITED','ABANDONED'):raise EvidenceError('production process has no terminal observation; preserve without completion')
    terminal=supervision['terminal'];terminal_status(terminal,expected_job)
    terminal_name='exit.json' if terminal['state']=='EXITED' else 'abandoned.json'
    path='jobs/'+expected_job+'/'+terminal_name
    if observe(control,path,output/'terminal-before.json',65536,deadline)!=terminal:
        raise EvidenceError('observed terminal evidence differs')
    retained=[]
    for index,(transport,name,absolute) in enumerate(selected):
        snapshot=output/f'root-{index:03d}'
        receipt=export(transport,name,store,snapshot,deadline,progress=progress,whole_root=name=='.')
        retained.append({'declared_root':absolute,'profile_sha256':digest(transport.profile),
                         'receipt_path':str((snapshot/'export.json').resolve()),'receipt_sha256':digest(receipt)})
    if observe(control,path,output/'terminal-after.json',65536,deadline)!=terminal:
        raise EvidenceError('terminal evidence changed during retention')
    first=read_json(Path(retained[0]['receipt_path']))
    if read_json(Path(first['files_directory'])/terminal_name)!=terminal:raise EvidenceError('exported terminal differs')
    result={'schema':'ovl.production-terminal-retention.v1','job_sha256':expected_job,'worker_sha256':expected_worker,
            'pod_id':control.profile['pod_id'],'terminal':terminal,'roots':retained,
            'scope':'all declared outputs copied after observed terminal status, including all existing primary/recovery/partial bytes',
            'training_replay':'NOT_RUN','production_acceptance':'NOT_RUN'}
    write_json(output/'retention.json',result);return result


def verify(value,control,transports,job_file,expected_job,expected_worker):
    fields(value,'schema job_sha256 worker_sha256 pod_id terminal roots scope training_replay production_acceptance','terminal retention')
    job=read_json(Path(job_file))
    if (digest(job)!=expected_job or value['schema']!='ovl.production-terminal-retention.v1' or value['job_sha256']!=expected_job
        or value['worker_sha256']!=expected_worker or value['pod_id']!=control.profile['pod_id']
        or value['training_replay']!='NOT_RUN' or value['production_acceptance']!='NOT_RUN'):
        raise EvidenceError('retention identity changed')
    terminal_status(value['terminal'],expected_job)
    selected=roots(control,transports,job,expected_job)
    if len(value['roots'])!=len(selected):raise EvidenceError('declared production output omitted')
    receipts=[]
    for item,(transport,name,absolute) in zip(value['roots'],selected):
        fields(item,'declared_root profile_sha256 receipt_path receipt_sha256','retained root')
        if item['declared_root']!=absolute or item['profile_sha256']!=digest(transport.profile):raise EvidenceError('retained root selection differs')
        path=Path(item['receipt_path'])
        if path.is_symlink():raise EvidenceError('retained receipt symlink')
        receipt=read_json(path)
        if (digest(receipt)!=item['receipt_sha256'] or receipt['schema']!='ovl.offpod-versioned-tree-export.v1'
            or receipt['pod_id']!=value['pod_id'] or receipt['profile_sha256']!=item['profile_sha256'] or receipt['root']!=name
            or receipt['result']!='PASS' or receipt['numerical_verification']!='NOT_RUN'):
            raise EvidenceError('retained byte receipt differs')
        directory=Path(receipt['files_directory'])
        if directory.is_symlink():raise EvidenceError('retained files symlink')
        verify_inventory(directory,receipt['files']);receipts.append(receipt)
    name='exit.json' if value['terminal']['state']=='EXITED' else 'abandoned.json'
    if read_json(Path(receipts[0]['files_directory'])/name)!=value['terminal']:raise EvidenceError('retained terminal record differs')
    return receipts
