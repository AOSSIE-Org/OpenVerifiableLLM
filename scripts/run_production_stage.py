"""Supervise one caller-authenticated production job on an existing rental.

The enclosing coordinator must verify the production packet, independent publisher
policies, full replay parents and cost admission before selecting this job. This
adapter is not an admission gate. Numerical drivers reverify their public parents.
It preserves the original launch fence, all output roots and fixed stop deadlines.
"""
from pathlib import Path
import time
import uuid

from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer
from pod_job_client import LAUNCH_CONTROL_SECONDS,launch,reconcile_launch,save_once,job_supervision,worker_stop_request,stop_delivery
from pod_observation_retry import read as bounded_read
from production_health import ProductionHealth
from production_checkpoint_poll import CheckpointRetention
from production_boundary_poll import BoundaryPublisher
from production_retention import retain,verify
from workload_health import terminal_status


def stop_window(output,requested,job,plan,now,export_seconds):
    """Keep the first stop and accept one later immutable controller cause.

    Callers validate the current controller identity before reaching this helper.
    A later cause can shorten the hard-stop bound, never renew it. The original
    numerical stop payload remains unchanged even when a second cause appears.
    """
    def hard(request):
        started=request.get('observed_epoch',plan['request_checkpoint_epoch'])
        integer(started,plan['input']['now_epoch'],now,'stop request clock')
        return min(job['deadline_epoch'],started+plan['input']['checkpoint_grace_seconds']
                   -export_seconds-job['stop_grace_seconds'])
    path=output/'stop-intent.json'
    if requested is None:
        if path.exists():raise EvidenceError('retained production stop disappeared')
        return None,None
    if not path.exists():save_once(path,{'request':requested,'hard_stop_epoch':hard(requested)})
    first=read_json(path);fields(first,'request hard_stop_epoch','production stop intent')
    if first['hard_stop_epoch']!=hard(first['request']):raise EvidenceError('original hard stop changed')
    later=output/'controller-stop-intent.json'
    if first['request']!=requested:
        graceful={'schema':'ovl.dispatcher-stop-request.v1','job_sha256':digest(job),'reason':'fixed graceful stop'}
        if first['request']!=graceful or requested.get('schema')!='ovl.rental-stop-request.v1':
            raise EvidenceError('stop request changed outside graceful-to-controller transition')
        value={'request':requested,'first_stop_sha256':digest(first),
               'hard_stop_epoch':min(first['hard_stop_epoch'],hard(requested))}
        save_once(later,value)
        return first['request'],value['hard_stop_epoch']
    if later.exists():raise EvidenceError('retained later controller stop disappeared')
    return first['request'],first['hard_stop_epoch']


def run_stage(control,health,job_file,expected_job,worker_file,expected_worker,output,health_file,
              stop_file,rental_intent_sha256,checkpoint,publisher,*,export_seconds,sleep=time.sleep):
    """Live checkpoint/publication hooks plus complete terminal retention.

    Completion means retained bytes after observed process exit, including failed
    and partial outputs. It never means successful training or replay. Only the
    enclosing coordinator may finish the rental after checking all planned work.
    """
    require_digest(expected_job);require_digest(expected_worker);require_digest(rental_intent_sha256)
    job=read_json(Path(job_file))
    if digest(job)!=expected_job or job.get('kind') not in ('production-record','full-replay'):
        raise EvidenceError('exact production job required')
    if Path(worker_file).is_symlink() or file_hash(Path(worker_file))!=expected_worker:
        raise EvidenceError('production worker differs from selection')
    from production_run_health import ProductionRunHealth
    if type(health) not in (ProductionHealth,ProductionRunHealth):raise EvidenceError('production retention health required')
    binding=health.bindings.get(expected_job)
    if (binding is None or binding['control'] is not control or Path(binding['job_file'])!=Path(job_file)
        or binding['worker_sha256']!=expected_worker):raise EvidenceError('stage differs from health binding')
    if (type(checkpoint) is not CheckpointRetention or checkpoint.job!=expected_job
        or checkpoint.health is not health or checkpoint.health_file!=health_file):
        raise EvidenceError('selected live checkpoint hook required')
    if job['kind']=='production-record':
        if (type(publisher) is not BoundaryPublisher or publisher.job!=expected_job
            or publisher.health is not health or publisher.health_file!=health_file
            or publisher.transport is not checkpoint.transport):
            raise EvidenceError('selected public boundary hook required')
    elif publisher is not None:raise EvidenceError('full replay cannot publish new recorded boundaries')
    contract=health.contract(expected_job)
    integer(export_seconds,1,1800,'terminal export interval')
    if export_seconds+job['stop_grace_seconds']>health.plan['input']['checkpoint_grace_seconds']:
        raise EvidenceError('stop grace leaves insufficient terminal export reserve')
    integer(job['deadline_epoch'],health.plan['input']['now_epoch']+1,
            health.plan['provider_terminate_epoch']-export_seconds-job['stop_grace_seconds'],
            'production worker deadline leaving terminal export reserve')
    activity=job['environment'].get('OVL_ACTIVITY_FILE')
    activity_transport=None;activity_name=None
    if activity is not None:
        candidates=[t for t in [control,*binding['transports']] if activity.startswith(t.profile['remote_root']+'/')]
        if len(candidates)!=1:raise EvidenceError('activity lacks a unique selected transport')
        activity_transport=candidates[0];activity_name=activity[len(activity_transport.profile['remote_root'])+1:]
        if activity_transport is control and activity_name.startswith('jobs/'):
            raise EvidenceError('activity overlaps worker control records')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    identity={'schema':'ovl.production-stage-selection.v1','job_sha256':expected_job,
              'worker_sha256':expected_worker,'contract_sha256':digest(contract),
              'rental_intent_sha256':rental_intent_sha256,'export_seconds':export_seconds,
              'checkpoint_selection_sha256':digest(checkpoint.identity),
              'publisher_selection_sha256':digest(publisher.identity) if publisher else None}
    save_once(output/'selection.json',identity)
    fence=output/'launch/launch-intent.json';result_file=output/'stage-result.json'
    if not fence.exists() and not result_file.exists() and (
        Path(stop_file).exists() or health.now()>=min(job['deadline_epoch'],health.plan['request_checkpoint_epoch'])):
        raise EvidenceError('stop or expired deadline forbids new production launch')
    health.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':expected_job,
                      'pod_id':health.pod,'kind':job['kind']})

    def complete(proof_path):
        proof=read_json(proof_path)
        verify(proof,control,binding['transports'],job_file,expected_job,expected_worker)
        result={'schema':'ovl.production-stage-result.v1','job_sha256':expected_job,
                'selection_sha256':digest(identity),'retention_path':str(proof_path.resolve()),
                'retention_sha256':digest(proof),'terminal':proof['terminal'],
                'scope':'complete declared output retention after observed exit; numerical acceptance NOT_RUN'}
        save_once(result_file,result)
        if not health.jobs[expected_job]['finished']:
            health.terminal_retained(expected_job,proof_path)
            (health.job_exit if proof['terminal']['state']=='EXITED' else health.abandon_job)(expected_job,proof['terminal'])
        else:health.retained(expected_job)
        health.write(health_file);return result

    if result_file.exists():
        result=read_json(result_file)
        return complete(Path(result['retention_path']))
    if health.jobs[expected_job]['finished']:raise EvidenceError('finished production job lacks retained stage result')
    job_root='jobs/'+expected_job
    def deliver_stop(now):
        requested=None
        if Path(stop_file).exists():
            requested=read_json(Path(stop_file))
            fields(requested,'schema intent_sha256 pod_id observed_epoch reasons','controller stop request')
            if (requested['schema']!='ovl.rental-stop-request.v1' or requested['intent_sha256']!=rental_intent_sha256
                or requested['pod_id']!=health.pod):raise EvidenceError('foreign production stop request')
        elif now>=health.plan['request_checkpoint_epoch']:
            requested={'schema':'ovl.dispatcher-stop-request.v1','job_sha256':expected_job,'reason':'fixed graceful stop'}
        # Recording gets a numerical checkpoint stop first. The already selected
        # worker deadline remains the hard process bound for both record/replay.
        first,hard=stop_window(output,requested,job,health.plan,now,export_seconds)
        if requested is not None:
            marker=output/'request-stop';save_once(marker,first)
            if publisher is not None:
                stop_delivery(checkpoint.transport,expected_job,'request-stop',marker,output/'stop-delivery.json',
                              min(health.plan['external_terminate_epoch'],health.now()+30))
            if publisher is None or health.now()>=hard:
                worker_marker=output/'worker-stop.json';save_once(worker_marker,worker_stop_request(expected_job))
                stop_delivery(control,expected_job,job_root+'/request-stop',worker_marker,output/'worker-stop-delivery.json',
                              min(health.plan['external_terminate_epoch'],health.now()+30))
        return requested

    if fence.exists():
        selected={'schema':'ovl.offpod-job-launch-intent.v1','job_sha256':expected_job,
                  'worker_sha256':expected_worker,'profile_sha256':digest(control.profile)}
        if read_json(fence)!=selected:raise EvidenceError('retained production launch selection changed')
        requested=deliver_stop(health.now())
        limit=min(health.plan['external_terminate_epoch'],health.now()+LAUNCH_CONTROL_SECONDS)
        if requested is not None and publisher is not None and not(output/'worker-stop-delivery.json').exists():
            _,hard=stop_window(output,requested,job,health.plan,health.now(),export_seconds)
            limit=min(limit,hard)
        reconcile_launch(control,selected,output/'launch',limit)
        deliver_stop(health.now())
    else:
        def before_start():
            if Path(stop_file).exists() or health.now()>=min(job['deadline_epoch'],health.plan['request_checkpoint_epoch']):
                raise EvidenceError('stop or expired deadline before production launch fence')
        launch(control,job_file,expected_job,worker_file,expected_worker,output/'launch',
               min(job['deadline_epoch'],health.plan['request_checkpoint_epoch'],health.now()+LAUNCH_CONTROL_SECONDS),before_start=before_start)
    while True:
        health.write(health_file);now=health.now()
        if now>=health.plan['external_terminate_epoch']:raise EvidenceError('external production rental deadline reached')
        requested=deliver_stop(now)
        observation=output/'observations'/uuid.uuid4().hex;observation.mkdir(mode=0o700,parents=True)
        supervision=bounded_read('supervision',control,(expected_job,expected_worker),health,health_file,observation,sleep=sleep,
                                 private_diagnostics=output/'private-transport-diagnostics')
        write_json(observation/'supervision.json',supervision)
        if supervision['state'] in ('SUPERVISOR_ABSENT','LAUNCH_FENCE_WITHOUT_INTENT'):
            terminal=job_supervision(control,expected_job,expected_worker,min(health.plan['external_terminate_epoch'],health.now()+30),abandon=True)
            write_json(observation/'abandonment.json',terminal);terminal_status(terminal,expected_job);break
        if supervision['state'] in ('EXITED','ABANDONED'):
            terminal=supervision['terminal'];terminal_status(terminal,expected_job)
            if supervision['child_alive']:raise EvidenceError('terminal observation contradicts live child')
            if not supervision['runner_alive']:break
        if supervision['state'] in ('CHILD_IDENTITY_UNKNOWN','LAUNCH_NOT_OBSERVED'):
            raise EvidenceError('production process identity unresolved; retain guards and evidence')
        values=bounded_read('metadata',control,{job_root+'/input-progress.json':observation/'input-progress.json'},
                            health,health_file,observation,sleep=sleep,private_diagnostics=output/'private-transport-diagnostics')
        progress=values[job_root+'/input-progress.json']
        if progress is not None:
            fields(progress,'schema job_sha256 verified_bytes','production input progress')
            prefixes=[];count=0
            for item in job['required_files']:count+=item['bytes'];prefixes.append(count)
            if (progress['schema']!='ovl.job-input-progress.v1' or progress['job_sha256']!=expected_job
                or type(progress['verified_bytes']) is not int or progress['verified_bytes'] not in prefixes):
                raise EvidenceError('production input observation differs from selected inventory prefix')
            health.bytes(digest({'job':expected_job,'operation':'remote-verified-input-prefix'}),
                         {'bytes_sent':0,'bytes_received':progress['verified_bytes']},total=count)
        if activity is not None:
            values=bounded_read('metadata',activity_transport,{activity_name:observation/'activity.json'},health,health_file,observation,sleep=sleep,
                                private_diagnostics=output/'private-transport-diagnostics')
            if values[activity_name] is not None:health.activity(expected_job,values[activity_name])
        retained=checkpoint.poll()
        if publisher is not None and requested is None:
            publication=publisher.poll(retained_selection=None if retained is None else retained['selection'])
            if publication is not None and publication.get('schema')=='ovl.production-boundary-completion.v1':
                from consolidate_boundary_storage import consolidate
                consolidate(publisher.output,publication,checkpoint.store)
        health.write(health_file);sleep(5)
    intent_file=output/'terminal-export-intent.json'
    if not intent_file.exists():
        started=health.now()
        save_once(intent_file,{'schema':'ovl.production-stage-export.v1','selection_sha256':digest(identity),
                  'terminal':terminal,'started_epoch':started,
                  'deadline_epoch':min(health.plan['external_terminate_epoch'],started+export_seconds)})
    intent=read_json(intent_file)
    fields(intent,'schema selection_sha256 terminal started_epoch deadline_epoch','production terminal export intent')
    if (intent['schema']!='ovl.production-stage-export.v1' or intent['selection_sha256']!=digest(identity)
        or intent['terminal']!=terminal or intent['deadline_epoch']!=min(health.plan['external_terminate_epoch'],intent['started_epoch']+export_seconds)):
        raise EvidenceError('terminal export identity/deadline changed')
    integer(intent['started_epoch'],health.plan['input']['now_epoch'],health.plan['external_terminate_epoch'],'terminal export start')
    destination=output/'terminal-export'
    if destination.exists() and not(destination/'retention.json').exists():destination=output/'terminal-export-retry'
    if destination.exists() and not(destination/'retention.json').exists():raise EvidenceError('terminal export retries exhausted; preserve both copies')
    if not destination.exists():
        if health.now()>=intent['deadline_epoch']:raise EvidenceError('original terminal export deadline expired')
        def progress(operation,counts,total):health.bytes(operation,counts,total=total);health.write(health_file)
        retain(control,binding['transports'],job_file,expected_job,expected_worker,checkpoint.store,
               destination,intent['deadline_epoch'],progress=progress)
    if read_json(destination/'retention.json')['terminal']!=terminal:raise EvidenceError('retained terminal differs from observed exit')
    return complete(destination/'retention.json')
