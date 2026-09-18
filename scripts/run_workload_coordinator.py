#!/usr/bin/env python3
"""Persistent finite workload dispatch inside an existing guarded rental.

Never provisions, extends a deadline or admits production. Pin the finite plan
independently; run with restart-on-failure under the external rental guards.
"""
import argparse
from pathlib import Path
import time

from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,inventory,read_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.supervision import Journal
from pod_job_client import save_once
from pod_transfer import Transport,relative
from run_rental_controller import validate as validate_rental,watchdog_heartbeat
from run_workload_stage import run_stage,saved_result
from workload_health import Health


def validate(plan,expected,rental,transport,inputs,worker):
    require_digest(expected)
    if digest(plan)!=expected:raise EvidenceError('workload plan differs from independent selection')
    fields(plan,'schema rental_intent_sha256 profile_sha256 worker_sha256 timing uploads stages','finite workload plan')
    if plan['schema']!='ovl.finite-workload-plan.v3':raise EvidenceError('unsupported finite workload plan')
    w,p=validate_rental(rental,plan['rental_intent_sha256'])
    timing=plan['timing'];fields(timing,'export_reserve_seconds transfer_floor_bytes_per_second hash_floor_bytes_per_second basis','finite timing budget')
    integer(timing['export_reserve_seconds'],60,1800,'finite export reserve')
    integer(timing['transfer_floor_bytes_per_second'],65536,2**30,'selected transfer planning floor')
    integer(timing['hash_floor_bytes_per_second'],65536,2**34,'selected hashing planning floor')
    if type(timing['basis']) is not str or not 1<=len(timing['basis'])<=1024:raise EvidenceError('explicit timing basis required; estimates are not measurements')
    if digest(transport.profile)!=plan['profile_sha256']:raise EvidenceError('workload endpoint differs from selected plan')
    require_digest(plan['worker_sha256']);worker=Path(worker)
    if worker.is_symlink() or file_hash(worker)!=plan['worker_sha256']:raise EvidenceError('workload worker source differs')
    uploads=plan['uploads'];targets=set();total=0
    if type(uploads) is not list or len(uploads)>100000:raise EvidenceError('bounded selected upload inventory required')
    for upload in uploads:
        fields(upload,'path remote_path bytes sha256','selected input upload')
        relative(upload['path']);relative(upload['remote_path']);require_digest(upload['sha256'])
        integer(upload['bytes'],0,2**40,'selected input bytes');total+=upload['bytes']
        if upload['bytes']>180*timing['transfer_floor_bytes_per_second']:
            raise EvidenceError('individual upload exceeds bounded retry window')
        if not upload['remote_path'].startswith('inputs/') or upload['remote_path'] in targets:
            raise EvidenceError('uploads must use distinct immutable input paths')
        targets.add(upload['remote_path'])
        local=confined(Path(inputs),upload['path'])
        if not local.is_file() or local.stat().st_size!=upload['bytes']:raise EvidenceError('selected upload missing or wrong size')
    if total>2**40:raise EvidenceError('selected upload inventory too large')
    if type(plan['stages']) is not list or not 1<=len(plan['stages'])<=32:raise EvidenceError('bounded explicit stage list required')
    jobs=[];names=set();hashes=set();setup_count=0
    for stage in plan['stages']:
        fields(stage,'name job_path job_sha256 maximum_export_bytes','selected finite stage')
        integer(stage['maximum_export_bytes'],1,2**40,'selected finite output size bound')
        relative(stage['name']);relative(stage['job_path']);require_digest(stage['job_sha256'])
        if '/' in stage['name'] or stage['name'] in names or stage['job_sha256'] in hashes:raise EvidenceError('duplicate or nested stage identity')
        names.add(stage['name']);hashes.add(stage['job_sha256'])
        path=confined(Path(inputs),stage['job_path']);job=read_json(path)
        if digest(job)!=stage['job_sha256']:raise EvidenceError('selected stage descriptor changed')
        if job.get('kind') not in ('setup','pilot','export'):raise EvidenceError('dedicated production/replay hooks required')
        from pod_job_worker import validate_job,Refusal
        from run_workload_stage import remote_name
        try:validate_job(job,resolve_executable=False)
        except Refusal as error:raise EvidenceError('invalid selected job descriptor: '+str(error)) from None
        if job['kind']=='setup':
            setup_count+=1
            if setup_count>1 or jobs or job['argv'][:3]!=['/usr/bin/python3','-I','-S']:
                raise EvidenceError('only one initial isolated image-Python setup stage is permitted')
        reserve=timing['export_reserve_seconds'];floor=timing['transfer_floor_bytes_per_second'];hash_floor=timing['hash_floor_bytes_per_second']
        size=stage['maximum_export_bytes']
        needed=(size+floor-1)//floor+(6*size+hash_floor-1)//hash_floor+30
        if needed>reserve:raise EvidenceError('export reserve does not cover selected output transfer and six hash passes')
        if job['stop_grace_seconds']+reserve>p['input']['checkpoint_grace_seconds']:
            raise EvidenceError('worker stop grace leaves insufficient export reserve')
        if not p['input']['now_epoch']<job['deadline_epoch']<=p['input']['now_epoch']+1500 or job['deadline_epoch']+reserve>min(p['request_checkpoint_epoch'],p['input']['now_epoch']+1800):
            raise EvidenceError('finite plan misses reserved stop/export-age window')
        selected_roots=['jobs/'+stage['job_sha256'],*[remote_name(transport,path) for path in job['export_roots']]]
        if len(set(selected_roots))!=len(selected_roots) or any(a!=b and b.startswith(a+'/') for a in selected_roots for b in selected_roots):
            raise EvidenceError('overlapping selected finite export roots')
        if any(name in ('inputs','tools') or name.startswith(('inputs/','tools/')) for name in selected_roots):
            raise EvidenceError('finite output overlaps immutable setup input namespace')
        if 'OVL_ACTIVITY_FILE' in job['environment']:remote_name(transport,job['environment']['OVL_ACTIVITY_FILE'])
        required={f['path']:f for f in job['required_files']}
        for upload in uploads:
            target=transport.profile['remote_root']+'/'+upload['remote_path']
            if required.get(target)!={'path':target,'bytes':upload['bytes'],'sha256':upload['sha256']}:
                raise EvidenceError('every selected upload must be rehashed before each finite stage')
        jobs.append((stage,path,job))
    return w,p,jobs


def controller_observation(directory,rental,pod,*,starting):
    """Read one complete immutable journal prefix without contending for its lease."""
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():raise EvidenceError('existing rental journal required')
    events=Journal(directory)._read()
    if [e['body'] for e in events if e['kind']=='creation-intent']!=[rental]:
        raise EvidenceError('rental journal does not select this exact intent')
    known={e['body']['id'] for e in events if e['kind']=='creation-observed'}
    if known!={pod}:raise EvidenceError('rental has not uniquely adopted selected pod')
    stopped=any(e['kind']=='teardown' and e['body'].get('complete') is True for e in events)
    stopping=any(e['kind']=='decision' and e['body'].get('action') in ('CHECKPOINT_AND_STOP','TERMINATE') for e in events)
    if starting and (stopped or stopping):raise EvidenceError('rental is stopping; no new workload')
    return {'journal_prefix_sha256':digest(events[-1]),'pod_id':pod,'stopping':stopping,'terminated':stopped}


def retain_observation(output,observation):
    directory=Path(output)/'controller-observations';directory.mkdir(mode=0o700,exist_ok=True)
    save_once(directory/(digest(observation)+'.json'),observation)


def check_export_budget(stage,result):
    # A planning bound is not a filesystem quota. Preserve oversized exports,
    # refuse to advance, and let the existing independent guards stop billing.
    if sum(f['bytes'] for e in result['exports'] for f in e['files'])>stage['maximum_export_bytes']:
        raise EvidenceError('retained finite output exceeded selected export budget; no next stage')


def finalize(health,plan,expected,output,result,health_file,stop):
    """Adopt a prepared final report only after reconstructing its exact inputs."""
    fields(result,'schema plan_sha256 pod_id outcome stages unstarted_stages scope provider_termination production_acceptance','finite result')
    if (result['schema']!='ovl.finite-workload-result.v1' or result['plan_sha256']!=expected or result['pod_id']!=health.pod
        or result['provider_termination']!='SEPARATE_CONTROLLER_REQUIRED' or result['production_acceptance']!='NOT_RUN'):
        raise EvidenceError('saved finite result identity changed')
    history=result['stages']
    if type(history) is not list or not 1<=len(history)<=len(plan['stages']):raise EvidenceError('invalid completed stage prefix')
    for entry,stage in zip(history,plan['stages']):
        path=output/'stages'/stage['name']/'stage-result.json'
        actual=saved_result(path,stage['job_sha256'])
        check_export_budget(stage,actual)
        selected={'name':stage['name'],'job_sha256':stage['job_sha256'],'result_path':str(path.resolve()),
                  'result_sha256':file_hash(path),'exit':actual['exit']}
        if entry!=selected:raise EvidenceError('saved completed stage differs from retained exports')
    if set(health.jobs)!={s['job_sha256'] for s in history}:raise EvidenceError('final result omits a launched job')
    if result['unstarted_stages']!=[s['name'] for s in plan['stages'][len(history):]]:raise EvidenceError('unstarted stage selection changed')
    failed=[i for i,e in enumerate(history) if e['exit']['state']!='EXITED' or e['exit']['exit_code']!=0]
    if failed:
        if failed!=[len(history)-1] or result['outcome']!='FAILED_OR_ABANDONED':raise EvidenceError('stage failure outcome changed')
    elif result['outcome']=='STOPPED_AFTER_STAGE':
        if not stop.exists():raise EvidenceError('saved stop has no retained controller marker')
    elif result['outcome']!='EXITED_ZERO' or len(history)!=len(plan['stages']):raise EvidenceError('incomplete plan reported as exited zero')
    final=output/'final';names=['result.json']
    for index,entry in enumerate(history):
        dest=final/f'stage-{index:03d}';dest.mkdir(mode=0o700,exist_ok=True)
        name='exit.json' if entry['exit']['state']=='EXITED' else 'abandoned.json'
        save_once(dest/name,entry['exit']);names.append(f'stage-{index:03d}/{name}')
    health.finish(final,inventory(final,names));health.write(health_file)
    return result


def run(plan,expected,rental,controller_directory,watchdog_file,transport,inputs,worker,output,health_file,*,sleep=time.sleep):
    w,p,jobs=validate(plan,expected,rental,transport,inputs,worker)
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    with Journal(output/'health-journal').lease() as journal:
        selected=output/'selected-plan.json'
        if journal.events and not selected.exists():raise EvidenceError('existing health has no bound coordinator plan')
        save_once(selected,plan)
        health=Health(journal,w,transport.profile['pod_id'])
        if health.complete:
            # Read-only post-deadline adoption: do not emit a new live heartbeat.
            files=read_json(journal.directory/'final-export-inventory.json')
            final=[e['body']['detail'] for e in journal.events if e['body'].get('kind')=='complete']
            if len(final)!=1 or digest(files)!=final[0]['export_inventory_sha256']:raise EvidenceError('completed coordinator export selection changed')
            if Path(final[0]['directory'])!=(output/'final').resolve():raise EvidenceError('completed coordinator directory moved; refuse unverified copied result')
            health._check_final(Path(final[0]['directory']),files)
            # A crash may have followed finish() before its health write. Restore
            # that completion while the original lifetime is still live, so the
            # provider controller can tear down promptly. Never extend lifetime.
            if health.lifetime.remaining()>0:health.write(health_file)
            return read_json(output/'final/result.json')
        history=[];outcome='EXITED_ZERO';stop=Path(controller_directory)/'stop-request.json'
        if (output/'final/result.json').exists():
            return finalize(health,plan,expected,output,read_json(output/'final/result.json'),health_file,stop)
        upload_dir=output/'uploads';upload_dir.mkdir(mode=0o700,exist_ok=True)
        for upload in plan['uploads']:
            identity=digest({'profile_sha256':plan['profile_sha256'],'upload':upload})
            receipt=upload_dir/(identity+'.json')
            if receipt.exists():
                saved=read_json(receipt)
                if saved.get('selection')!=upload or saved.get('operation_sha256')!=identity:
                    raise EvidenceError('saved upload selection changed')
                # The pinned worker rehashes selected required inputs before use.
                continue
            if health.jobs:raise EvidenceError('launched job lacks prior durable input transfer receipt')
            retain_observation(output,controller_observation(controller_directory,rental,health.pod,starting=True))
            watchdog_heartbeat(watchdog_file,w,health.now(),health.pod)
            if stop.exists() or health.now()>=p['request_checkpoint_epoch']:raise EvidenceError('stop forbids new setup transfers')
            local=confined(Path(inputs),upload['path'])
            if file_hash(local)!=upload['sha256']:raise EvidenceError('selected input upload bytes changed')
            def progress(counts):
                health.bytes(identity,counts,total=upload['bytes'],direction='send');health.write(health_file)
            seconds=(upload['bytes']+plan['timing']['transfer_floor_bytes_per_second']-1)//plan['timing']['transfer_floor_bytes_per_second']+30
            transfer=transport.put(upload['remote_path'],local,min(p['request_checkpoint_epoch'],health.now()+seconds),progress=progress)
            if transfer['sha256']!=upload['sha256'] or transfer['bytes_sent']!=upload['bytes']:
                raise EvidenceError('actual uploaded input differs from selected inventory')
            save_once(receipt,{'selection':upload,'operation_sha256':identity,'transfer':transfer})
        for stage,path,job in jobs:
            started=stage['job_sha256'] in health.jobs
            observation=controller_observation(controller_directory,rental,health.pod,starting=not started)
            retain_observation(output,observation)
            if observation['terminated']:raise EvidenceError('rental terminated before required export; preserve and reconcile')
            if not started:watchdog_heartbeat(watchdog_file,w,health.now(),health.pod)
            if not started and job['deadline_epoch']+plan['timing']['export_reserve_seconds']>health.exported+1800:
                raise EvidenceError('actual export age leaves insufficient next-stage reserve')
            health.write(health_file)
            stage_output=output/'stages'/stage['name']
            result=run_stage(transport,health,path,stage['job_sha256'],worker,plan['worker_sha256'],stage_output,
                             health_file,stop,plan['rental_intent_sha256'],sleep=sleep)
            # Re-read retained exports, including terminal records, before advancing.
            if saved_result(stage_output/'stage-result.json',stage['job_sha256'])!=result:
                raise EvidenceError('retained stage result differs')
            check_export_budget(stage,result)
            history.append({'name':stage['name'],'job_sha256':stage['job_sha256'],'result_path':str((stage_output/'stage-result.json').resolve()),
                            'result_sha256':file_hash(stage_output/'stage-result.json'),'exit':result['exit']})
            if result['exit']['state']!='EXITED' or result['exit']['exit_code']!=0:
                outcome='FAILED_OR_ABANDONED';break
            if stop.exists():outcome='STOPPED_AFTER_STAGE';break
        final=output/'final';final.mkdir(mode=0o700,exist_ok=True)
        result={'schema':'ovl.finite-workload-result.v1','plan_sha256':expected,'pod_id':health.pod,'outcome':outcome,
                'stages':history,'unstarted_stages':[s['name'] for s,_,_ in jobs[len(history):]],
                'scope':'finite selected job exits and complete selected off-pod exports; no numerical acceptance',
                'provider_termination':'SEPARATE_CONTROLLER_REQUIRED','production_acceptance':'NOT_RUN'}
        save_once(final/'result.json',result)
        return finalize(health,plan,expected,output,result,health_file,stop)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','rental-intent','controller-journal','watchdog-heartbeat','profile','key','known-hosts','inputs','worker','output','health'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--plan-sha256',required=True);a=parser.parse_args()
    try:
        transport=Transport(read_json(a.profile),a.key,a.known_hosts)
        result=run(read_json(a.plan),a.plan_sha256,read_json(a.rental_intent),a.controller_journal,a.watchdog_heartbeat,
                   transport,a.inputs,a.worker,a.output,a.health)
        print(result['schema']+' '+digest(result))
    except Exception as error:parser.exit(1,'finite coordinator refused: '+type(error).__name__+'; guards remain responsible, preserve evidence\n')


if __name__=='__main__':main()
