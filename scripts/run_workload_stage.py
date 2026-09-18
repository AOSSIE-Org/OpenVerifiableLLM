"""One bounded pilot/setup stage within an already guarded, adopted rental.

Called by a persistent workload coordinator holding the Health journal lease.
This does not create compute or authorize production; production recording and
long full replay need their checkpoint hooks, and are explicitly rejected here.
"""
from pathlib import Path
import time
import uuid

from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer
from pod_checkpoint_handoff import observe
from pod_job_client import launch,export_tree,job_supervision
from pod_transfer import relative
from workload_health import terminal_status


def remote_name(transport,path):
    prefix=transport.profile['remote_root']+'/'
    if type(path) is not str or not path.startswith(prefix):raise EvidenceError('job path outside independently selected remote root')
    return relative(path.removeprefix(prefix))


def saved_result(path,job):
    value=read_json(path)
    fields(value,'schema job_sha256 exit exports scope','workload stage result')
    if value['schema']!='ovl.workload-stage-result.v1' or value['job_sha256']!=job:raise EvidenceError('wrong saved stage identity')
    for item in value['exports']:
        fields(item,'remote_root directory files','saved stage export')
        directory=Path(item['directory'])
        if directory.is_symlink():raise EvidenceError('saved export directory is a symlink')
        verify_inventory(directory,item['files'])
    exit_record=value['exit']
    terminal_status(exit_record,job)
    return value


def run_stage(transport,health,job_file,expected_job,worker_file,expected_worker,output,health_file,
              stop_file,rental_intent_sha256,*,sleep=time.sleep):
    """Resume by pinned identity, export all selected roots, retain exact failures.

    Health completion is deliberately left to the enclosing coordinator after
    all planned stages exit and its final export is verified. The finite stage
    cap provides time for export before the controller's 1800-second export age.
    """
    require_digest(rental_intent_sha256);job=read_json(Path(job_file));require_digest(expected_job)
    if digest(job)!=expected_job:raise EvidenceError('stage job differs from operator selection')
    if job.get('kind') not in ('setup','pilot','export'):
        raise EvidenceError('production/long replay requires dedicated verified checkpoint hooks')
    now=health.now();deadline=job['deadline_epoch']
    integer(deadline,health.plan['input']['now_epoch']+1,health.plan['provider_terminate_epoch'],'stage deadline')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    stage_file=output/'stage-result.json'
    if expected_job not in health.jobs:
        if now>=health.plan['request_checkpoint_epoch'] or not 0<deadline-now<=1500:
            raise EvidenceError('new stage exceeds bounded work/export interval or graceful-stop deadline')
    job_root='jobs/'+expected_job
    # Job records are added by the coordinator. Embedding jobs/<job hash> in the
    # job itself would create an impossible self-referential digest dependency.
    roots=[job_root,*[remote_name(transport,p) for p in job['export_roots']]]
    if len(set(roots))!=len(roots):raise EvidenceError('duplicate export roots')
    # Nested roots waste copies and can hide the final terminal record selection.
    if any(a!=b and b.startswith(a+'/') for a in roots for b in roots):raise EvidenceError('overlapping export roots')
    activity=remote_name(transport,job['environment']['OVL_ACTIVITY_FILE']) if 'OVL_ACTIVITY_FILE' in job['environment'] else None
    health.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':expected_job,'pod_id':transport.profile['pod_id'],'kind':job['kind']})
    if stage_file.exists():
        result=saved_result(stage_file,expected_job)
        if not health.jobs[expected_job]['finished']:
            (health.job_exit if result['exit']['state']=='EXITED' else health.abandon_job)(expected_job,result['exit'])
        health.write(health_file);return result
    if health.jobs[expected_job]['finished']:raise EvidenceError('finished stage lacks preserved result; do not run again')
    if not(output/'launch/launch-intent.json').exists() and (Path(stop_file).exists() or health.now()>=health.plan['request_checkpoint_epoch']):
        raise EvidenceError('stop requested before launch; no new workload may start')
    # First transfer deadline remains before the fixed external rental deadline.
    launch(transport,job_file,expected_job,worker_file,expected_worker,output/'launch',min(deadline,int(health.wall())+60))
    iterations=output/'observations';iterations.mkdir(mode=0o700,exist_ok=True)
    stop_delivered=(output/'stop-delivery.json').exists()
    while True:
        health.write(health_file);now=health.now()
        observation=iterations/uuid.uuid4().hex;observation.mkdir(mode=0o700)
        transfer_deadline=min(health.plan['external_terminate_epoch'],now+30)
        requested=None
        if Path(stop_file).exists():
            requested=read_json(Path(stop_file))
            fields(requested,'schema intent_sha256 pod_id observed_epoch reasons','controller stop request')
            if requested['schema']!='ovl.rental-stop-request.v1' or requested['intent_sha256']!=rental_intent_sha256 or requested['pod_id']!=health.pod:
                raise EvidenceError('unrelated controller stop request')
        elif now>=health.plan['request_checkpoint_epoch']:
            requested={'schema':'ovl.dispatcher-stop-request.v1','job_sha256':expected_job,'reason':'fixed graceful-stop deadline'}
        if requested is not None and not stop_delivered:
            marker=output/'request-stop';write_json(marker,requested)
            receipt=transport.put(job_root+'/request-stop',marker,transfer_deadline)
            write_json(output/'stop-delivery.json',receipt);stop_delivered=True
        supervision=job_supervision(transport,expected_job,expected_worker,transfer_deadline)
        write_json(observation/'supervision.json',supervision)
        if supervision['state'] in ('SUPERVISOR_ABSENT','LAUNCH_FENCE_WITHOUT_INTENT'):
            abandoned=job_supervision(transport,expected_job,expected_worker,transfer_deadline,abandon=True)
            write_json(observation/'abandonment-response.json',abandoned)
            terminal_status(abandoned,expected_job)
            exit_value=abandoned;break
        if supervision['state']=='ABANDONED':
            exit_value=supervision['terminal'];terminal_status(exit_value,expected_job);break
        if supervision['state'] in ('CHILD_IDENTITY_UNKNOWN','LAUNCH_NOT_OBSERVED'):
            raise EvidenceError('unobservable job identity; external guard and preserved exports required')
        exit_value=observe(transport,job_root+'/exit.json',observation/'exit.json',65536,transfer_deadline,optional=True)
        status=observe(transport,job_root+'/status.json',observation/'status.json',65536,transfer_deadline,optional=True)
        if exit_value is not None:
            fields(exit_value,'schema job_sha256 state exit_code','workload terminal observation')
            integer(exit_value['exit_code'],-255,255,'workload exit code')
            if exit_value['schema']!='ovl.workload-job-exit.v1' or exit_value['job_sha256']!=expected_job or exit_value['state']!='EXITED':
                raise EvidenceError('foreign or malformed workload exit')
            failure=observe(transport,job_root+'/failure.json',observation/'failure.json',65536,transfer_deadline,optional=True)
            if (status is not None and status.get('job_sha256')==expected_job and status.get('state')=='EXITED') or failure is not None:
                break
        progress=observe(transport,job_root+'/input-progress.json',observation/'input-progress.json',65536,transfer_deadline,optional=True)
        if progress is not None:
            fields(progress,'schema job_sha256 verified_bytes','input verification progress')
            prefixes=[];count=0
            for item in job['required_files']:count+=item['bytes'];prefixes.append(count)
            if progress['schema']!='ovl.job-input-progress.v1' or progress['job_sha256']!=expected_job or type(progress['verified_bytes']) is not int or progress['verified_bytes'] not in prefixes:
                raise EvidenceError('input progress differs from selected inventory prefix')
            # Selected worker's hash-read counters, not a network or truth proof.
            health.bytes(digest({'job':expected_job,'operation':'remote-verified-input-prefix'}),{'bytes_sent':0,'bytes_received':progress['verified_bytes']},total=count)
        if activity is not None:
            value=observe(transport,activity,observation/'activity.json',65536,transfer_deadline,optional=True)
            if value is not None:health.activity(expected_job,value)
        health.write(health_file);sleep(5)
    exports=[]
    for index,name in enumerate(roots):
        destination=output/f'export-{index:03d}'
        if destination.exists():
            receipt=read_json(destination/'export.json')
            if receipt.get('root')!=name or receipt.get('pod_id')!=health.pod or receipt.get('result')!='PASS':
                raise EvidenceError('incomplete/foreign preserved export; inspect without replacing it')
            verify_inventory(destination/'files',receipt['files'])
        else:
            def progress(operation,counts,total):health.bytes(operation,counts,total=total);health.write(health_file)
            receipt=export_tree(transport,name,destination,health.plan['external_terminate_epoch'],progress=progress)
        health.exported_files(expected_job,destination/'files',receipt['files'])
        exports.append({'remote_root':name,'directory':str((destination/'files').resolve()),'files':receipt['files']})
        health.write(health_file)
    selected=next(e for e in exports if e['remote_root']==job_root)
    terminal_name='exit.json' if exit_value['state']=='EXITED' else 'abandoned.json'
    if read_json(Path(selected['directory'])/terminal_name)!=exit_value:raise EvidenceError('exported terminal status differs from observed exit')
    result={'schema':'ovl.workload-stage-result.v1','job_sha256':expected_job,'exit':exit_value,'exports':exports,
            'scope':'observed finite stage exit and complete selected off-pod bytes; no production/replay acceptance'}
    write_json(stage_file,result)
    (health.job_exit if exit_value['state']=='EXITED' else health.abandon_job)(expected_job,exit_value)
    health.write(health_file)
    return result
