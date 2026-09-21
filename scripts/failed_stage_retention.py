"""One finite backup of an exhausted development stage, with no success credit.

The normal checkpoint retry gate remains closed. A separate intent fences this
retention-only operation and its original deadlines; interruption cannot create
another attempt. The external rental guards keep termination authority.
"""
from pathlib import Path
import time

from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,verify_inventory
from ovl_pipeline.schema import fields,integer
from pod_job_client import save_once
from pod_transfer import RangeRecoveryExhausted,retryable_transport
from pod_versioned_export import export,regular_directory
from preserved_downloads import observations,compare
from run_workload_stage import remote_name,saved_result
from workload_health import terminal_status


def classify(error,failure,transport,output,now):
    """Only positively classified exhaustion can select the backup path."""
    if type(error) is not RangeRecoveryExhausted or not retryable_transport(error.__cause__):return
    if hasattr(error,'transport_cleanup_diagnostic'):return
    save_once(Path(output)/'failure-retention-eligibility.json',{
        'schema':'ovl.failed-stage-retention-eligibility.v1','failure_sha256':digest(failure),
        'profile_sha256':digest(transport.profile),'job_sha256':failure['job_sha256'],
        'observed_epoch':now,'cause':'classified-transient-range-exhaustion',
        'qualification':'FAILED','scope':'one separate terminal backup; no checkpoint retry or acknowledgement'})


def prepare(transport,health,job_file,job_root,worker,worker_root,output,stop_file,rental_root,limits,store):
    output=Path(output);job=read_json(Path(job_file));failure=read_json(output/'dispatch-failure.json')
    if digest(job)!=job_root or job.get('kind')!='pilot' or file_hash(Path(worker))!=worker_root:
        raise EvidenceError('failed-stage backup identity differs')
    launch=read_json(output/'launch/launch-intent.json')
    if launch!={'schema':'ovl.offpod-job-launch-intent.v1','job_sha256':job_root,
                'worker_sha256':worker_root,'profile_sha256':digest(transport.profile)}:
        raise EvidenceError('failed-stage backup launch identity differs')
    eligible=read_json(output/'failure-retention-eligibility.json')
    fields(eligible,'schema failure_sha256 profile_sha256 job_sha256 observed_epoch cause qualification scope','failure backup eligibility')
    if (eligible['schema']!='ovl.failed-stage-retention-eligibility.v1' or eligible['failure_sha256']!=digest(failure)
        or eligible['profile_sha256']!=digest(transport.profile) or eligible['job_sha256']!=job_root
        or eligible['cause']!='classified-transient-range-exhaustion' or eligible['qualification']!='FAILED'
        or failure.get('error_type')!='RangeRecoveryExhausted' or failure.get('job_sha256')!=job_root):
        raise EvidenceError('failed-stage backup classification differs')
    fields(limits,'maximum_bytes maximum_uncached_bytes export_seconds','failure backup bounds')
    integer(limits['maximum_bytes'],1,2**40,'failure backup size')
    integer(limits['maximum_uncached_bytes'],1,limits['maximum_bytes'],'failure backup uncached size')
    integer(limits['export_seconds'],1,1800,'failure backup interval')
    integer(eligible['observed_epoch'],health.plan['input']['now_epoch'],health.now(),'failure observation clock')
    roots=['jobs/'+job_root,*[remote_name(transport,p) for p in job['export_roots']]]
    if any(a==b or a.startswith(b+'/') or b.startswith(a+'/') for i,a in enumerate(roots) for b in roots[i+1:]):
        raise EvidenceError('overlapping failed-stage backup roots')
    # The fixed stage reserve and first failure's shutdown allowance are both
    # ceilings. A later controller stop can shorten them, never renew them.
    deadline=min(health.plan['external_terminate_epoch'],job['deadline_epoch']+limits['export_seconds'],
                 eligible['observed_epoch']+min(1800,health.plan['input']['checkpoint_grace_seconds']))
    base=regular_directory(output/'failure-retention');intent_file=base/'intent.json'
    binding={'job_sha256':job_root,'worker_sha256':worker_root,'profile_sha256':digest(transport.profile),
             'failure_sha256':digest(failure),'eligibility_sha256':digest(eligible),'rental_intent_sha256':rental_root,
             'roots':roots,'store':str(Path(store).absolute()),'limits':limits,'deadline_epoch':deadline}
    if intent_file.exists():
        intent=read_json(intent_file)
        if intent.get('binding')!=binding:raise EvidenceError('failed-stage backup intent changed')
    else:
        intent={'schema':'ovl.failed-stage-retention-intent.v1','binding':binding,
                'prior_observations':observations(store,roots,transport.profile)}
        save_once(intent_file,intent)
    if intent.get('schema')!='ovl.failed-stage-retention-intent.v1':raise EvidenceError('wrong failed-stage backup intent')
    mono=getattr(transport,'monotonic',time.monotonic);end=mono()+max(0,deadline-health.now())
    def check():
        if digest(transport.profile)!=binding['profile_sha256'] or read_json(intent_file)!=intent:
            raise EvidenceError('failure backup identity changed during retention')
        if digest(read_json(output/'dispatch-failure.json'))!=binding['failure_sha256']:
            raise EvidenceError('original dispatch failure changed during retention')
        bound=deadline
        if Path(stop_file).exists():
            stop=read_json(Path(stop_file));fields(stop,'schema intent_sha256 pod_id observed_epoch reasons','failure backup stop')
            if stop['schema']!='ovl.rental-stop-request.v1' or stop['intent_sha256']!=rental_root or stop['pod_id']!=health.pod:
                raise EvidenceError('foreign failure backup stop')
            integer(stop['observed_epoch'],health.plan['input']['now_epoch'],health.now(),'controller stop clock')
            save_once(base/'controller-stop.json',stop)
            bound=min(bound,stop['observed_epoch']+health.plan['input']['checkpoint_grace_seconds'])
        elif (base/'controller-stop.json').exists():raise EvidenceError('failure backup controller stop disappeared')
        if health.now()>=bound or mono()>=end:raise EvidenceError('fixed failure backup deadline expired')
        return bound
    return intent,check


def retained(output,job_root,profile):
    output=Path(output);base=output/'failure-retention';intent=read_json(base/'intent.json')
    failure=read_json(output/'dispatch-failure.json');eligible=read_json(output/'failure-retention-eligibility.json')
    b=intent['binding']
    if (b['job_sha256']!=job_root or b['profile_sha256']!=digest(profile) or b['failure_sha256']!=digest(failure)
        or b['eligibility_sha256']!=digest(eligible) or eligible['qualification']!='FAILED'):
        raise EvidenceError('retained failure backup identity changed')
    result=saved_result(base/'stage-result.json',job_root)
    if read_json(base/'terminal.json')!=result['exit']:raise EvidenceError('retained failure backup terminal changed')
    if read_json(output/'terminal-observation.json')!=result['exit']:raise EvidenceError('first terminal observation changed')
    proof=read_json(base/'retention.json')
    if proof!={'schema':'ovl.failed-stage-retention.v1','intent_sha256':digest(intent),
               'result_sha256':digest(result),'qualification':'FAILED','numerical_verification':'NOT_RUN'}:
        raise EvidenceError('failed-stage backup proof changed')
    if [e['remote_root'] for e in result['exports']]!=b['roots']:raise EvidenceError('failed-stage backup roots changed')
    compare(intent['prior_observations'],result['exports'],lambda:None)
    return result


def retain(transport,health,job_root,output,health_file,intent,check,terminal):
    output=Path(output);base=output/'failure-retention';binding=intent['binding']
    terminal_status(terminal,job_root)
    from run_workload_stage import retain_terminal
    retain_terminal(output,job_root,terminal)
    save_once(base/'terminal.json',terminal)
    if (base/'retention.json').exists():result=retained(output,job_root,transport.profile)
    else:
        check();exports=[];remaining=binding['limits']['maximum_bytes'];uncached=binding['limits']['maximum_uncached_bytes']
        # One backup attempt per root. Completed roots are rehashed on restart;
        # any incomplete root is preserved and cannot reset the finite attempt.
        for index,name in enumerate(binding['roots']):
            destination=base/f'export-{index:03d}'
            if destination.exists():
                if not(destination/'returned.json').exists():raise EvidenceError('failure backup attempt interrupted; no automatic reset')
                receipt=read_json(destination/'export.json')
                returned=read_json(destination/'returned.json')
                if returned!={'intent_sha256':digest(intent),'receipt_sha256':digest(receipt)}:
                    raise EvidenceError('failure backup returned export changed')
                if (receipt.get('root')!=name or receipt.get('profile_sha256')!=digest(transport.profile)
                    or receipt.get('result')!='PASS' or receipt.get('files_directory')!=str(destination/'files')):
                    raise EvidenceError('failure backup export identity changed')
                verify_inventory(destination/'files',receipt['files'])
            else:
                def progress(operation,counts,total):
                    check();health.bytes(operation,counts,total=total);health.write(health_file)
                receipt=export(transport,name,binding['store'],destination,check(),maximum_bytes=remaining,
                               maximum_uncached_bytes=uncached,progress=progress,allow_bulk=False)
                check();save_once(destination/'returned.json',{'intent_sha256':digest(intent),'receipt_sha256':digest(receipt)})
            check();remaining-=sum(item['bytes'] for item in receipt['files'])
            uncached-=receipt['bounds']['selected_missing_bytes']
            if min(remaining,uncached)<0:raise EvidenceError('failure backup size allowance exceeded')
            exports.append({'remote_root':name,'directory':str((destination/'files').resolve()),'files':receipt['files']})
        compare(intent['prior_observations'],exports,check)
        name='exit.json' if terminal['state']=='EXITED' else 'abandoned.json'
        if read_json(Path(exports[0]['directory'])/name)!=terminal:raise EvidenceError('failure backup terminal changed')
        result={'schema':'ovl.workload-stage-result.v1','job_sha256':job_root,'exit':terminal,'exports':exports,
                'scope':'failed-stage retention only; qualification FAILED; no successor or numerical acceptance'}
        check();save_once(base/'stage-result.json',result)
        save_once(base/'retention.json',{'schema':'ovl.failed-stage-retention.v1','intent_sha256':digest(intent),
                  'result_sha256':digest(result),'qualification':'FAILED','numerical_verification':'NOT_RUN'})
    if result['exit']!=terminal:raise EvidenceError('failure backup terminal differs from live observation')
    if not health.jobs[job_root]['finished']:
        for entry in result['exports']:
            health.exported_files(job_root,entry['directory'],entry['files'],deadline=check())
        (health.job_exit if terminal['state']=='EXITED' else health.abandon_job)(job_root,terminal)
        health.write(health_file)
    return result
