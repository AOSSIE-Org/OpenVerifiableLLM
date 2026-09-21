"""Stop one already-fenced failed stage and retain its real terminal outputs.

No repeated launch or fabricated terminal result. Rental/watchdog deadlines remain
authoritative; failure to observe the owned process fails closed with all local
partial evidence preserved.
"""
from pathlib import Path
import time

from ovl_pipeline.canonical import EvidenceError,read_json
from pod_job_client import job_supervision,save_once,worker_stop_request,stop_delivery
from run_workload_stage import run_stage,retain_terminal


def stop_and_retain(transport,health,job_file,job_root,worker,worker_root,output,health_file,stop_file,rental_root,*,sleep=time.sleep,initial_retention=None,terminal_limits=None,failure_limits=None):
    output=Path(output)
    if (output/'failure-retention-eligibility.json').exists():
        fatal=output/'failure-retention-failure.json'
        if fatal.exists():raise EvidenceError('preserved failure-retention refusal; no automatic retry')
        try:
            return _stop_and_retain(transport,health,job_file,job_root,worker,worker_root,output,health_file,stop_file,rental_root,
                sleep=sleep,initial_retention=initial_retention,terminal_limits=terminal_limits,failure_limits=failure_limits)
        except Exception as error:
            save_once(fatal,{'schema':'ovl.failed-stage-backup-failure.v1','job_sha256':job_root,'error_type':type(error).__name__,
                            'scope':'strict retained refusal; no retry or qualification credit'})
            raise
    return _stop_and_retain(transport,health,job_file,job_root,worker,worker_root,output,health_file,stop_file,rental_root,
        sleep=sleep,initial_retention=initial_retention,terminal_limits=terminal_limits,failure_limits=failure_limits)


def _stop_and_retain(transport,health,job_file,job_root,worker,worker_root,output,health_file,stop_file,rental_root,*,sleep,initial_retention,terminal_limits,failure_limits):
    output=Path(output)
    if not(output/'launch/launch-intent.json').exists():raise EvidenceError('no fenced stage to stop; never invent a launch')
    recovery=None
    if (output/'failure-retention-eligibility.json').exists():
        from failed_stage_retention import prepare,retained
        if failure_limits is None or initial_retention is None:
            raise EvidenceError('failed-stage backup requires selected bounds and object store')
        recovery=prepare(transport,health,job_file,job_root,worker,worker_root,output,stop_file,rental_root,
                         failure_limits,initial_retention.store)
        if health.jobs.get(job_root,{}).get('finished'):return retained(output,job_root,transport.profile)
    if health.jobs.get(job_root,{}).get('finished'):
        return run_stage(transport,health,job_file,job_root,worker,worker_root,output,health_file,stop_file,rental_root,
                         sleep=sleep,initial_retention=initial_retention,terminal_limits=terminal_limits)
    marker=output/'dispatch-stop.json'
    save_once(marker,{'schema':'ovl.sustained-dispatch-stop.v1','job_sha256':job_root,
                      'reason':'local dispatcher failure; stop and preserve all declared outputs'})
    worker_marker=output/'worker-stop.json'
    save_once(worker_marker,worker_stop_request(job_root))
    deadline=lambda:min(recovery[1]() if recovery else health.plan['external_terminate_epoch'],health.now()+30)
    receipt=output/'dispatch-stop-delivery.json'
    if receipt.exists():
        stop_delivery(transport,job_root,'jobs/'+job_root+'/request-stop',worker_marker,receipt,deadline())
    else:
        state=job_supervision(transport,job_root,worker_root,deadline())
        if state['state'] in ('EXITED','ABANDONED'):retain_terminal(output,job_root,state['terminal'])
        if state['state'] not in ('EXITED','ABANDONED'):
            # Immutable put accepts exact existing bytes without replacement;
            # foreign or changed stop content remains a strict failure.
            stop_delivery(transport,job_root,'jobs/'+job_root+'/request-stop',worker_marker,receipt,deadline())
    # Observe only immutable worker/process supervision while shutting down. A bad
    # activity file cannot trap recovery before the complete terminal export.
    while True:
        health.write(health_file)
        value=job_supervision(transport,job_root,worker_root,deadline())
        if value['state'] in ('EXITED','ABANDONED'):retain_terminal(output,job_root,value['terminal'])
        if value['state']=='EXITED' and not value['child_alive'] and not value['runner_alive']:break
        if value['state']=='ABANDONED':break
        if value['state'] in ('SUPERVISOR_ABSENT','LAUNCH_FENCE_WITHOUT_INTENT'):
            abandoned=job_supervision(transport,job_root,worker_root,deadline(),abandon=True)
            retain_terminal(output,job_root,abandoned);continue
        if value['state'] in ('CHILD_IDENTITY_UNKNOWN','LAUNCH_NOT_OBSERVED'):
            raise EvidenceError('failed stage identity remains unobservable; external guards retain authority')
        sleep(1)
    # The existing stage path adopts the old fence and immediately observes the
    # actual terminal worker record. It cannot launch another numerical process.
    if recovery is not None:
        from failed_stage_retention import retain
        return retain(transport,health,job_root,output,health_file,*recovery,value['terminal'])
    return run_stage(transport,health,job_file,job_root,worker,worker_root,output,health_file,stop_file,rental_root,
                     sleep=sleep,initial_retention=initial_retention,terminal_limits=terminal_limits)
