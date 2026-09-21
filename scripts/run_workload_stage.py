"""One bounded pilot/setup stage within an already guarded, adopted rental.

Called by a persistent workload coordinator holding the Health journal lease.
This does not create compute or authorize production; production recording and
long full replay need their checkpoint hooks, and are explicitly rejected here.
"""
from pathlib import Path
import time
import uuid

from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer
from pod_observe import observe_many
from pod_job_client import LAUNCH_CONTROL_SECONDS,launch,reconcile_launch,export_tree,job_supervision,save_once,worker_stop_request,stop_delivery,retain_stop_reason
from pod_transfer import relative
from workload_health import terminal_status
from pod_observation_retry import read as bounded_read


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
              stop_file,rental_intent_sha256,*,sleep=time.sleep,initial_retention=None,terminal_limits=None):
    """Resume by pinned identity, export all selected roots, retain exact failures.

    Health completion is deliberately left to the enclosing coordinator after
    all planned stages exit and its final export is verified. The finite stage
    cap provides time for export before the controller's 1800-second export age.
    """
    require_digest(rental_intent_sha256);job=read_json(Path(job_file));require_digest(expected_job)
    if digest(job)!=expected_job:raise EvidenceError('stage job differs from operator selection')
    require_digest(expected_worker)
    if Path(worker_file).is_symlink() or file_hash(Path(worker_file))!=expected_worker:
        raise EvidenceError('stage worker differs from operator selection')
    if job.get('kind') not in ('setup','pilot','export'):
        raise EvidenceError('production/long replay requires dedicated verified checkpoint hooks')
    if initial_retention is not None:
        from pilot_retention import InitialRetention
        from pilot_checkpoint_retention import CheckpointRetention
        if (type(initial_retention) not in (InitialRetention,CheckpointRetention) or initial_retention.job!=expected_job
            or initial_retention.health is not health or initial_retention.transport is not transport
            or initial_retention.health_file!=health_file or job['kind']!='pilot'):
            raise EvidenceError('initial retention differs from selected pilot stage')
    if terminal_limits is not None:
        from pilot_checkpoint_retention import CheckpointRetention
        if type(initial_retention) is not CheckpointRetention:raise EvidenceError('terminal reuse limits require verified checkpoint delivery')
        fields(terminal_limits,'maximum_bytes maximum_uncached_bytes','terminal export limits')
        for k in terminal_limits:integer(terminal_limits[k],1,2**40,k)
        if terminal_limits['maximum_uncached_bytes']!=initial_retention.selection['maximum_uncached_export_bytes']:
            raise EvidenceError('terminal uncached limit differs from selected retention')
    now=health.now();deadline=job['deadline_epoch']
    integer(deadline,health.plan['input']['now_epoch']+1,health.plan['provider_terminate_epoch'],'stage deadline')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    stage_file=output/'stage-result.json'
    launch_intent=output/'launch/launch-intent.json'
    if not stage_file.exists() and not launch_intent.exists():
        maximum_work=2100 if initial_retention is not None else 1500
        if now>=health.plan['request_checkpoint_epoch'] or not 0<deadline-now<=maximum_work:
            marker=output/'unlaunched-stage.json'
            if not marker.exists():write_json(marker,{'schema':'ovl.unlaunched-stage-refusal.v1','job_sha256':expected_job,
                'observed_epoch':now,'reason':'compute/graceful deadline before launch fence','execution':'NOT_STARTED_BY_THIS_COORDINATOR',
                'scope':'local launch fence absent; no invented remote terminal or health completion'})
            raise EvidenceError('new stage exceeds bounded work/export interval or graceful-stop deadline')
    job_root='jobs/'+expected_job
    # Job records are added by the coordinator. Embedding jobs/<job hash> in the
    # job itself would create an impossible self-referential digest dependency.
    roots=[job_root,*[remote_name(transport,p) for p in job['export_roots']]]
    if len(set(roots))!=len(roots):raise EvidenceError('duplicate export roots')
    # Nested roots waste copies and can hide the final terminal record selection.
    if any(a!=b and b.startswith(a+'/') for a in roots for b in roots):raise EvidenceError('overlapping export roots')
    activity=remote_name(transport,job['environment']['OVL_ACTIVITY_FILE']) if 'OVL_ACTIVITY_FILE' in job['environment'] else None
    if activity is not None and activity.startswith('jobs/'):
        raise EvidenceError('activity must not overlap worker-owned job metadata')
    health.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':expected_job,'pod_id':transport.profile['pod_id'],'kind':job['kind']})
    if stage_file.exists():
        result=saved_result(stage_file,expected_job)
        if not health.jobs[expected_job]['finished']:
            (health.job_exit if result['exit']['state']=='EXITED' else health.abandon_job)(expected_job,result['exit'])
        health.write(health_file);return result
    if health.jobs[expected_job]['finished']:raise EvidenceError('finished stage lacks preserved result; do not run again')
    if not(output/'launch/launch-intent.json').exists() and (Path(stop_file).exists() or health.now()>=health.plan['request_checkpoint_epoch']):
        raise EvidenceError('stop requested before launch; no new workload may start')
    def deliver_stop(now):
        requested=None
        if Path(stop_file).exists():
            requested=read_json(Path(stop_file))
            fields(requested,'schema intent_sha256 pod_id observed_epoch reasons','controller stop request')
            if requested['schema']!='ovl.rental-stop-request.v1' or requested['intent_sha256']!=rental_intent_sha256 or requested['pod_id']!=health.pod:
                raise EvidenceError('unrelated controller stop request')
        elif now>=health.plan['request_checkpoint_epoch']:
            requested={'schema':'ovl.dispatcher-stop-request.v1','job_sha256':expected_job,'reason':'fixed graceful-stop deadline'}
        retain_stop_reason(output,requested,expected_job,graceful_reason='fixed graceful-stop deadline')
        if requested is not None:
            marker=output/'request-stop';save_once(marker,worker_stop_request(expected_job))
            stop_delivery(transport,expected_job,job_root+'/request-stop',marker,output/'stop-delivery.json',
                          min(health.plan['external_terminate_epoch'],health.now()+30))


    # A retained launch fence permits only read-only adoption. Its observation
    # and evidence export must remain possible after the compute deadline; no
    # new workload child is authorized and the external rental deadline never
    # moves. Bounded observers, exporters and stop/abandon controls still run.
    if launch_intent.exists():
        selected={'schema':'ovl.offpod-job-launch-intent.v1','job_sha256':expected_job,
                  'worker_sha256':expected_worker,'profile_sha256':digest(transport.profile)}
        if read_json(launch_intent)!=selected:raise EvidenceError('retained launch identity differs')
        deliver_stop(health.now())
        reconcile_launch(transport,selected,output/'launch',min(health.plan['external_terminate_epoch'],int(health.wall())+LAUNCH_CONTROL_SECONDS))
    else:
        def before_start():
            if Path(stop_file).exists() or health.now()>=min(deadline,health.plan['request_checkpoint_epoch']):
                raise EvidenceError('stop or expired deadline before launch fence; no new workload may start')
        launch(transport,job_file,expected_job,worker_file,expected_worker,output/'launch',
               min(deadline,health.plan['request_checkpoint_epoch'],health.now()+LAUNCH_CONTROL_SECONDS),before_start=before_start)
    iterations=output/'observations';iterations.mkdir(mode=0o700,exist_ok=True)
    while True:
        health.write(health_file);now=health.now()
        observation=iterations/uuid.uuid4().hex;observation.mkdir(mode=0o700)
        def transfer_deadline():return min(health.plan['external_terminate_epoch'],health.now()+30)
        deliver_stop(now)
        supervision=bounded_read('supervision',transport,(expected_job,expected_worker),health,health_file,observation,sleep=sleep,
                                 private_diagnostics=output/'private-transport-diagnostics')
        write_json(observation/'supervision.json',supervision)
        if supervision['state'] in ('SUPERVISOR_ABSENT','LAUNCH_FENCE_WITHOUT_INTENT'):
            abandoned=job_supervision(transport,expected_job,expected_worker,transfer_deadline(),abandon=True)
            write_json(observation/'abandonment-response.json',abandoned)
            terminal_status(abandoned,expected_job)
            exit_value=abandoned;break
        if supervision['state']=='ABANDONED':
            exit_value=supervision['terminal'];terminal_status(exit_value,expected_job);break
        if supervision['state']=='EXITED':
            exit_value=supervision['terminal'];terminal_status(exit_value,expected_job)
            if supervision['child_alive']:raise EvidenceError('terminal record contradicts live workload child')
            # exit.json is durable before the final mutable status write. Wait
            # for the runner to leave, then export even if that write was lost.
            if not supervision['runner_alive']:break
        if supervision['state'] in ('CHILD_IDENTITY_UNKNOWN','LAUNCH_NOT_OBSERVED'):
            raise EvidenceError('unobservable job identity; external guard and preserved exports required')
        names=('exit.json','status.json','failure.json','input-progress.json')
        selected={job_root+'/'+name:observation/name for name in names}
        if activity is not None:selected[activity]=observation/'activity.json'
        if initial_retention is not None:
            for index,name in enumerate(initial_retention.observation_paths()):
                selected[name]=observation/f'initial-retention-{index:03d}.json'
        bundle=bounded_read('metadata',transport,selected,health,health_file,observation,sleep=sleep,
                            private_diagnostics=output/'private-transport-diagnostics')
        exit_value=bundle[job_root+'/exit.json'];status=bundle[job_root+'/status.json']
        if exit_value is not None:
            fields(exit_value,'schema job_sha256 state exit_code','workload terminal observation')
            integer(exit_value['exit_code'],-255,255,'workload exit code')
            if exit_value['schema']!='ovl.workload-job-exit.v1' or exit_value['job_sha256']!=expected_job or exit_value['state']!='EXITED':
                raise EvidenceError('foreign or malformed workload exit')
            failure=bundle[job_root+'/failure.json']
            if status is not None and type(status) is not dict:raise EvidenceError('invalid workload status object')
            if (status is not None and status.get('job_sha256')==expected_job and status.get('state')=='EXITED') or failure is not None:
                break
        progress=bundle[job_root+'/input-progress.json']
        if progress is not None:
            fields(progress,'schema job_sha256 verified_bytes','input verification progress')
            prefixes=[];count=0
            for item in job['required_files']:count+=item['bytes'];prefixes.append(count)
            if progress['schema']!='ovl.job-input-progress.v1' or progress['job_sha256']!=expected_job or type(progress['verified_bytes']) is not int or progress['verified_bytes'] not in prefixes:
                raise EvidenceError('input progress differs from selected inventory prefix')
            # Selected worker's hash-read counters, not a network or truth proof.
            health.bytes(digest({'job':expected_job,'operation':'remote-verified-input-prefix'}),{'bytes_sent':0,'bytes_received':progress['verified_bytes']},total=count)
        if activity is not None:
            value=bundle[activity]
            if value is not None:health.activity(expected_job,value)
        if initial_retention is not None:initial_retention.observe(bundle)
        health.write(health_file);sleep(5)
    exports=[];logical_bytes=0;uncached_bytes=0
    for index,name in enumerate(roots):
        from pod_versioned_export import classified_export,require_retryable_checkpoint
        bounds={} if terminal_limits is None else {'maximum_bytes':terminal_limits['maximum_bytes']-logical_bytes,
                    'maximum_uncached_bytes':terminal_limits['maximum_uncached_bytes']-uncached_bytes}
        destination=output/f'export-{index:03d}'
        # One fresh retry after a preserved incomplete transfer. A malformed
        # completed receipt is never replaced, and a second failure is terminal
        # for automatic retries. All incomplete directories remain untouched.
        if destination.exists() and not (destination/'export.json').exists():
            require_retryable_checkpoint(transport,name,destination,health.plan['external_terminate_epoch'],None,**bounds)
            preserved=destination;destination=output/f'export-{index:03d}-attempt-001'
            if destination.exists() and not (destination/'export.json').exists():
                raise EvidenceError('bounded export retries exhausted; preserve both attempts')
            marker=output/f'export-{index:03d}-recovery.json'
            if not marker.exists():write_json(marker,{'schema':'ovl.finite-export-recovery.v1','job_sha256':expected_job,
                'root':name,'preserved_incomplete_directory':str(preserved.resolve()),'selected_retry_directory':str(destination.resolve()),
                'original_external_deadline_epoch':health.plan['external_terminate_epoch'],'scope':'retained partial bytes; no integrity or computation credit'})
        if destination.exists():
            receipt=read_json(destination/'export.json')
            if receipt.get('root')!=name or receipt.get('pod_id')!=health.pod or receipt.get('result')!='PASS':
                raise EvidenceError('incomplete/foreign preserved export; inspect without replacing it')
            verify_inventory(destination/'files',receipt['files'])
        else:
            def progress(operation,counts,total):health.bytes(operation,counts,total=total);health.write(health_file)
            if initial_retention is None:
                receipt=classified_export(transport,name,destination,health.plan['external_terminate_epoch'],None,
                    lambda:export_tree(transport,name,destination,health.plan['external_terminate_epoch'],progress=progress))
            else:
                from pod_versioned_export import checkpoint_export
                receipt=checkpoint_export(transport,name,initial_retention.store,destination,health.plan['external_terminate_epoch'],progress=progress,expected_files=None,**bounds)
        if terminal_limits is not None:
            if receipt['schema']!='ovl.offpod-versioned-tree-export.v2':raise EvidenceError('bounded export receipt required')
            bound=receipt['bounds'];fields(bound,'maximum_bytes maximum_uncached_bytes selected_missing_bytes','retained export bounds')
            if bound['maximum_bytes']!=terminal_limits['maximum_bytes']-logical_bytes or bound['maximum_uncached_bytes']!=terminal_limits['maximum_uncached_bytes']-uncached_bytes:
                raise EvidenceError('retained terminal budgets differ')
            integer(bound['selected_missing_bytes'],0,bound['maximum_uncached_bytes'],'retained uncached bytes')
            logical_bytes+=sum(f['bytes'] for f in receipt['files']);uncached_bytes+=bound['selected_missing_bytes']
            if logical_bytes>terminal_limits['maximum_bytes']:raise EvidenceError('retained logical terminal bound exceeded')
        health.exported_files(expected_job,destination/'files',receipt['files'],deadline=health.plan['external_terminate_epoch'])
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
