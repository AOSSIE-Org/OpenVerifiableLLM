"""Freeze each sustained pilot job once, within its original rental plan.

Relative work budgets are resolved once at admission, never refreshed on restart.
Replay record roots and required bytes come only from completely retained parents.
The finite-v3 coordinator's absolute admission rules remain unchanged.
"""
from pathlib import Path

from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest
from ovl_pipeline.schema import fields,integer
from pod_job_client import save_once
from pod_job_worker import validate_job,Refusal


DEADLINE='@SELECTED_ORIGINAL_DEADLINE@'
RECORD='@RETAINED_RECORD_SHA256@'


def derive(stage,template,output,plan_sha256,rental_plan,now,*,parent=None,parent_kind='pilot'):
    """Return an immutable selected descriptor, including read-only adoption.

Caller validates the enclosing plan, template source, endpoint and parent policy;
this helper freezes only declared substitutions and complete parent required files.
"""
    require_digest(plan_sha256)
    parent_schemas={'pilot':'ovl.retained-pilot-record-parent.v1',
                    'initialization':'ovl.retained-initialization-record-parent.v1'}
    if parent_kind not in parent_schemas:raise EvidenceError('unsupported selected development parent kind')
    fields(stage,'name template_sha256 work_seconds export_reserve_seconds parent_record_root','sustained stage selection')
    if digest(template)!=stage['template_sha256']:raise EvidenceError('selected sustained template differs')
    integer(stage['work_seconds'],1,1500,'sustained stage work bound')
    integer(stage['export_reserve_seconds'],60,1800,'sustained final export reserve')
    if template.get('deadline_epoch')!=DEADLINE:raise EvidenceError('template must request one original selected deadline')
    if not isinstance(template.get('argv'),list):raise EvidenceError('explicit template argv required')
    if template.get('kind') not in ('setup','pilot','export'):raise EvidenceError('no production admission through development template')
    if template['stop_grace_seconds']+stage['export_reserve_seconds']>rental_plan['input']['checkpoint_grace_seconds']:
        raise EvidenceError('rental grace does not retain complete stage export reserve')
    if stage['parent_record_root'] is None:
        if parent is not None or RECORD in template['argv']:raise EvidenceError('undeclared replay parent')
    else:
        if (type(stage['parent_record_root']) is not str or not stage['parent_record_root'].startswith('/')
            or parent is None or parent.get('schema')!=parent_schemas[parent_kind]):
            raise EvidenceError('replay requires checked complete retained parent')
        if template['argv'].count(RECORD)!=1:raise EvidenceError('exactly one selected record digest argument required')
        require_digest(parent['record_sha256'])
        if digest(parent['record'])!=parent['record_sha256']:raise EvidenceError('checked record identity changed')
    output=Path(output)
    if any(p.is_symlink() for p in [output,*output.absolute().parents]):raise EvidenceError('sustained derivation directory symlink')
    output.mkdir(mode=0o700,parents=True,exist_ok=True)
    selected=output/'selection.json';job_file=output/'job.json'
    if job_file.exists() and not selected.exists():raise EvidenceError('existing job lost its original derivation fence; do not invent another')
    identity={'schema':'ovl.sustained-job-derivation.v1','plan_sha256':plan_sha256,'stage':stage,
              'template_sha256':digest(template),'parent_sha256':digest(parent) if parent else None}
    if parent_kind!='pilot':identity['parent_kind']=parent_kind
    if selected.exists():
        prior=read_json(selected)
        if prior.get('identity')!=identity:raise EvidenceError('retained sustained job derivation changed')
        admitted=prior['admitted_epoch'];deadline=prior['deadline_epoch']
    else:
        integer(now,rental_plan['input']['now_epoch'],rental_plan['request_checkpoint_epoch']-1,'stage admission clock')
        admitted=now
        deadline=min(now+stage['work_seconds'],rental_plan['request_checkpoint_epoch']-stage['export_reserve_seconds'])
        # Do not silently shorten a measured phase to squeeze it into leftovers.
        if deadline-now!=stage['work_seconds']:raise EvidenceError('remaining fixed rental cannot fit selected measured work/export budget')
    integer(admitted,rental_plan['input']['now_epoch'],rental_plan['request_checkpoint_epoch']-1,'retained admission clock')
    if deadline!=admitted+stage['work_seconds'] or deadline+stage['export_reserve_seconds']>rental_plan['request_checkpoint_epoch']:
        raise EvidenceError('retained phase deadline differs from original admission')
    job={**template,'deadline_epoch':deadline,'argv':[str(deadline) if a==DEADLINE else parent['record_sha256'] if a==RECORD else a for a in template['argv']]}
    if parent:
        additions=[{**f,'path':stage['parent_record_root']+'/'+f['path']} for f in parent['files']]
        job['required_files']=[*template['required_files'],*additions]
    try:validate_job(job,resolve_executable=False)
    except Refusal as error:raise EvidenceError('derived sustained job refused: '+str(error)) from None
    record={'identity':identity,'admitted_epoch':admitted,'deadline_epoch':deadline,'job_sha256':digest(job)}
    # Persist the derivation before the job file; restart repairs only this exact
    # missing file. Neither operation is a remote launch or launch retry.
    save_once(selected,record);save_once(job_file,job)
    return job_file,digest(job),job
