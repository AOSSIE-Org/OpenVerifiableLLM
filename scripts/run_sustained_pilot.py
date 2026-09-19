#!/usr/bin/env python3
"""Sequential development jobs inside one already guarded rental.

This dispatcher never creates compute, moves rental deadlines, grants production
admission or converts telemetry into verification. Derivations and launch fences
are adopted once. Complete output retention precedes every successor stage.
"""
from pathlib import Path
import time

from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,inventory,read_json,require_digest
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.supervision import Journal
from pod_job_client import save_once
from pod_transfer import relative
from sustained_health import SustainedHealth
from pilot_retention import InitialRetention
from pilot_checkpoint_retention import CheckpointRetention,selection as delivery_selection,command_selection
from pilot_record_parent import check as check_parent
from sustained_pilot_selection import derive,DEADLINE,RECORD
from run_rental_controller import validate as validate_rental,watchdog_heartbeat
from run_workload_coordinator import controller_observation,retain_observation,check_export_budget,journal_stop
from run_workload_stage import run_stage,saved_result,remote_name


def validate(plan,expected,rental,transport,inputs,worker):
    require_digest(expected)
    fields(plan,'schema rental_intent_sha256 profile_sha256 worker_sha256 timing uploads stages','sustained development plan')
    if digest(plan)!=expected or plan['schema'] not in ('ovl.sustained-pilot-plan.v1','ovl.initialization-cycle-plan.v1'):
        raise EvidenceError('selected sustained plan differs')
    initialization=plan['schema']=='ovl.initialization-cycle-plan.v1'
    watchdog,rental_plan=validate_rental(rental,plan['rental_intent_sha256'])
    if digest(transport.profile)!=plan['profile_sha256'] or file_hash(Path(worker))!=plan['worker_sha256']:
        raise EvidenceError('selected sustained endpoint/worker differs')
    timing=plan['timing'];fields(timing,'transfer_floor_bytes_per_second hash_floor_bytes_per_second basis','sustained timing')
    for k in ('transfer_floor_bytes_per_second','hash_floor_bytes_per_second'):integer(timing[k],65536,2**34,k)
    if type(timing['basis']) is not str or not 1<=len(timing['basis'])<=4096:raise EvidenceError('explicit measured timing basis required')
    if type(plan['uploads']) is not list or len(plan['uploads'])>100000:raise EvidenceError('bounded selected uploads required')
    uploaded={};total=0
    for u in plan['uploads']:
        fields(u,'path remote_path bytes sha256','sustained upload');relative(u['path']);relative(u['remote_path']);require_digest(u['sha256'])
        integer(u['bytes'],0,180*timing['transfer_floor_bytes_per_second'],'bounded upload retry window')
        if not u['remote_path'].startswith('inputs/') or u['remote_path'] in uploaded:raise EvidenceError('distinct immutable input upload paths required')
        local=confined(Path(inputs),u['path'])
        if not local.is_file() or local.stat().st_size!=u['bytes']:raise EvidenceError('selected upload missing/changed size')
        uploaded[u['remote_path']]=u;total+=u['bytes']
    if total>2**40:raise EvidenceError('sustained uploads too large')
    if type(plan['stages']) is not list or not 1<=len(plan['stages'])<=16:raise EvidenceError('bounded sustained stage list required')
    stages=[];prior={};prior_templates={};output_roots=[];initialization_actions=[]
    for s in plan['stages']:
        fields(s,'name template_path template_sha256 work_seconds export_reserve_seconds maximum_export_bytes parent_stage parent_record_root parent_binding validation_binding download_binding retention','sustained stage')
        relative(s['name']);relative(s['template_path']);require_digest(s['template_sha256'])
        if '/' in s['name'] or s['name'] in prior:raise EvidenceError('unique flat stage names required')
        integer(s['work_seconds'],1,1500 if initialization else 2100,'stage work seconds');integer(s['export_reserve_seconds'],60,1800,'stage export reserve')
        integer(s['maximum_export_bytes'],1,2**40,'complete stage export bound')
        size=s['maximum_export_bytes'];uncached=size
        if s['retention'] is not None and s['retention'].get('schema')=='ovl.pilot-checkpoint-retention.v1':
            delivery_selection(s['retention']);uncached=s['retention']['maximum_uncached_export_bytes']
            if uncached>size:raise EvidenceError('uncached terminal bound exceeds logical bound')
        needed=(uncached+timing['transfer_floor_bytes_per_second']-1)//timing['transfer_floor_bytes_per_second']+(10*size+timing['hash_floor_bytes_per_second']-1)//timing['hash_floor_bytes_per_second']+30
        if needed>s['export_reserve_seconds']:raise EvidenceError('complete terminal export budget lacks transfer/hash reserve')
        template=read_json(confined(Path(inputs),s['template_path']))
        if digest(template)!=s['template_sha256'] or template.get('deadline_epoch')!=DEADLINE:
            raise EvidenceError('immutable original-deadline template required')
        if template.get('kind') not in ('setup','pilot','export'):raise EvidenceError('sustained dispatcher cannot admit production')
        if template['kind']!='pilot' and s['work_seconds']>1500:
            raise EvidenceError('non-pilot stage exceeds original work bound')
        from pod_job_worker import validate_job,Refusal
        probe={**template,'deadline_epoch':rental_plan['input']['now_epoch']+s['work_seconds'],
               'argv':[str(rental_plan['input']['now_epoch']+s['work_seconds']) if a==DEADLINE else 'a'*64 if a==RECORD else a for a in template['argv']]}
        try:validate_job(probe,resolve_executable=False)
        except Refusal as error:raise EvidenceError('invalid sustained template: '+str(error)) from None
        if template['stop_grace_seconds']+s['export_reserve_seconds']>rental_plan['input']['checkpoint_grace_seconds']:
            raise EvidenceError('insufficient unchanged rental grace for full export')
        if s['parent_stage'] is None:
            if s['parent_record_root'] is not None or RECORD in template['argv']:raise EvidenceError('undeclared parent record')
        else:
            if s['parent_stage'] not in prior:
                raise EvidenceError('replay parent must be an earlier selected record stage')
            parent=prior[s['parent_stage']]
            if initialization:
                parent_argv=prior_templates[s['parent_stage']]['argv']
                parent_args=parent_argv[parent_argv.index('--')+1:] if '--' in parent_argv else []
                if (parent['validation_binding'] is None or parent['validation_binding'].get('action')!='record'
                    or parent_args.count('--output')!=1
                    or parent_args[parent_args.index('--output')+1:parent_args.index('--output')+2]!=[s['parent_record_root']]
                    or parent['parent_binding']!=s['parent_binding']):raise EvidenceError('initialization parent policy/root differs')
            elif (parent['retention'] is None or parent['retention']['mode']!='record'
                or parent['retention']['output_root']!=s['parent_record_root']
                or parent['parent_binding']!=s['parent_binding']):raise EvidenceError('replay parent policy/root differs')
        required={f['path']:f for f in template['required_files']}
        for name,u in uploaded.items():
            path=transport.profile['remote_root']+'/'+name
            if required.get(path)!={'path':path,'bytes':u['bytes'],'sha256':u['sha256']}:
                raise EvidenceError('each stage must rehash all uploaded input parents')
        roots=[remote_name(transport,p) for p in template['export_roots']]
        if not roots or any(r in ('inputs','tools','jobs') or r.startswith(('inputs/','tools/','jobs/')) for r in roots):
            raise EvidenceError('stage exports overlap immutable inputs or worker metadata')
        if any(a==b or a.startswith(b+'/') or b.startswith(a+'/') for a in roots for b in output_roots):
            raise EvidenceError('sustained stage outputs overlap another stage')
        if any(a!=b and a.startswith(b+'/') for a in roots for b in roots):raise EvidenceError('nested stage export roots')
        output_roots.extend(roots)
        if 'OVL_ACTIVITY_FILE' in template['environment']:
            activity=remote_name(transport,template['environment']['OVL_ACTIVITY_FILE'])
            if not any(activity.startswith(r+'/') for r in roots):raise EvidenceError('activity must be retained within a selected output root')
        if template['kind']=='pilot' and initialization:
            from initialization_parent import policy
            if (s['validation_binding'] is None or s['parent_binding'] is None or s['retention'] is not None
                or s['download_binding'] is not None or 'OVL_ACTIVITY_FILE' not in template['environment']):
                raise EvidenceError('initialization requires distinct scan/parent bindings and complete terminal retention')
            b=s['validation_binding'];fields(b,'schema stream_sha256 documents action','initialization validation binding')
            if b['schema']!='ovl.initialization-validation-binding.v1' or b['action'] not in ('record','verify'):
                raise EvidenceError('unsupported initialization action binding')
            require_digest(b['stream_sha256']);integer(b['documents'],1,2**53-1,'initialization document total')
            policy(s['parent_binding'])
            if b['stream_sha256']!=s['parent_binding']['stream_sha256']:raise EvidenceError('initialization stream parents differ')
            argv=template['argv']
            if argv.count('--')!=1 or argv[argv.index('--')+1:argv.index('--')+2]!=[b['action']]:
                raise EvidenceError('initialization action differs from selected bootstrap command')
            args=argv[argv.index('--')+1:]
            if (args.count('--output')!=1 or len(args)<=args.index('--output')+1
                or args[args.index('--output')+1] not in template['export_roots']):
                raise EvidenceError('initialization output must select one complete declared export root')
            if (b['action']=='verify')!=(s['parent_stage'] is not None):raise EvidenceError('initialization action has incorrect parent')
            if s['work_seconds']+s['export_reserve_seconds']>1800:
                raise EvidenceError('initialization must fit unchanged complete-export age')
            initialization_actions.append(b['action'])
        elif template['kind']=='pilot':
            if s['validation_binding'] is None or s['retention'] is None or s['parent_binding'] is None:
                raise EvidenceError('sustained numerical job requires exact stream and initial retention bindings')
            if s['download_binding'] is not None or 'OVL_ACTIVITY_FILE' not in template['environment']:
                raise EvidenceError('numerical pilot needs its own numerical activity protocol')
            binding=s['validation_binding'];fields(binding,'schema stream_sha256 documents','selected pilot validation binding')
            if binding['schema']!='ovl.pilot-validation-binding.v1':raise EvidenceError('wrong validation binding schema')
            require_digest(binding['stream_sha256']);integer(binding['documents'],1,2**53-1,'selected pilot documents')
            b=s['parent_binding'];fields(b,'schema recipe_sha256 kernel_sha256 stream_sha256 code_root','selected pilot record binding')
            if b['schema']!='ovl.pilot-record-parent-binding.v1':raise EvidenceError('wrong record binding schema')
            for key in ('recipe_sha256','kernel_sha256','stream_sha256','code_root'):require_digest(b[key])
            r=s['retention']
            if r.get('schema')=='ovl.pilot-checkpoint-retention.v1':
                delivery_selection(r);command_selection(template,r,DEADLINE)
                if r['binding']!=b:raise EvidenceError('checkpoint delivery parent differs')
                integer(r['maximum_checkpoint_bytes'],1,s['maximum_export_bytes'],'selected checkpoint bytes')
                needed_copy=(r['maximum_checkpoint_bytes']+timing['transfer_floor_bytes_per_second']-1)//timing['transfer_floor_bytes_per_second']+(10*r['maximum_checkpoint_bytes']+timing['hash_floor_bytes_per_second']-1)//timing['hash_floor_bytes_per_second']+30
                if needed_copy>r['copy_timeout_seconds']:raise EvidenceError('checkpoint copy budget lacks transfer/hash reserve')
            else:
                fields(r,'schema mode output_root phase maximum_initial_bytes','selected initial pilot retention')
                if r['schema']!='ovl.pilot-initial-retention.v1' or r['mode'] not in ('record','replay','resume') or r['phase'] not in ('wikipedia','conversation'):
                    raise EvidenceError('wrong selected initial pilot retention')
                integer(r['maximum_initial_bytes'],1,s['maximum_export_bytes'],'selected initial snapshot bytes')
            if s['retention']['output_root'] not in template['export_roots']:raise EvidenceError('initial checkpoint root not exported')
            if s['validation_binding']['stream_sha256']!=s['parent_binding']['stream_sha256']:
                raise EvidenceError('pilot validation and record parents differ')
            if s['retention']['mode']=='record' and s['parent_stage'] is not None:raise EvidenceError('record cannot load a replay parent')
            if s['retention']['mode']!='record' and s['parent_stage'] is None:raise EvidenceError('replay/resume has no record parent')
        elif any(s[k] is not None for k in ('validation_binding','retention','parent_binding','parent_stage')):
            raise EvidenceError('non-pilot must not carry numerical parent bindings')
        if s['download_binding'] is not None:
            b=s['download_binding'];fields(b,'schema plan_sha256 bytes','selected public download binding')
            if b['schema']!='ovl.public-input-binding.v1' or template['kind']!='setup' or 'OVL_ACTIVITY_FILE' not in template['environment']:
                raise EvidenceError('public input transfer needs a distinct observed setup job')
            require_digest(b['plan_sha256']);integer(b['bytes'],1,64*1024**3,'selected public input bytes')
        if template['kind']!='pilot' and s['work_seconds']+s['export_reserve_seconds']>1800:
            raise EvidenceError('uncheckpointed stage exceeds original export-age bound')
        prior[s['name']]=s;prior_templates[s['name']]=template;stages.append((s,template))
    if initialization and initialization_actions!=['record','verify']:
        raise EvidenceError('initialization cycle requires exactly one record followed by fresh regeneration')
    minimum=sum(s['work_seconds']+s['export_reserve_seconds'] for s,_ in stages)+(total+timing['transfer_floor_bytes_per_second']-1)//timing['transfer_floor_bytes_per_second']
    if minimum>rental_plan['request_checkpoint_epoch']-rental_plan['input']['now_epoch']:
        raise EvidenceError('complete selected phase budgets exceed unchanged rental work window')
    return watchdog,rental_plan,stages


def retained_stage(stage,output,profile):
    """Bind every retained root and terminal record to the selected descriptor."""
    from types import SimpleNamespace
    selected=read_json(output/'derived'/stage['name']/'selection.json')
    job=read_json(output/'derived'/stage['name']/'job.json');root=selected['job_sha256']
    if digest(job)!=root:raise EvidenceError('retained stage descriptor changed')
    result=saved_result(output/'stages'/stage['name']/'stage-result.json',root)
    expected=['jobs/'+root,*[remote_name(SimpleNamespace(profile=profile),p) for p in job['export_roots']]]
    if [e['remote_root'] for e in result['exports']]!=expected:
        raise EvidenceError('retained output roots differ from selected complete stage outputs')
    for e in result['exports']:
        directory=Path(e['directory']);actual=[]
        for p in directory.rglob('*'):
            if p.is_symlink() or not(p.is_file() or p.is_dir()):raise EvidenceError('unsafe retained stage output')
            if p.is_file():actual.append(p.relative_to(directory).as_posix())
        if sorted(actual)!=[f['path'] for f in e['files']]:raise EvidenceError('retained stage output inventory is incomplete')
    terminal='exit.json' if result['exit']['state']=='EXITED' else 'abandoned.json'
    if read_json(Path(result['exports'][0]['directory'])/terminal)!=result['exit']:
        raise EvidenceError('retained stage terminal differs from observed process result')
    return result


def parent_for(stage,stages,output,transport):
    if stage['parent_stage'] is None:return None
    parent_stage=next(s for s,_ in stages if s['name']==stage['parent_stage'])
    previous=retained_stage(parent_stage,output,transport.profile)
    if previous['exit']['state']!='EXITED' or previous['exit']['exit_code']!=0:raise EvidenceError('unsuccessful parent cannot launch replay')
    remote=remote_name(transport,stage['parent_record_root'])
    choices=[e for e in previous['exports'] if e['remote_root']==remote]
    if len(choices)!=1:raise EvidenceError('complete retained parent root absent')
    entry=choices[0]
    checker=check_parent
    if stage['parent_binding']['schema']=='ovl.initialization-record-parent-binding.v1':
        from initialization_parent import check as checker
    return checker(Path(entry['directory']),entry['files'],stage['parent_binding'])


def initialization_result(plan,output,profile):
    """Recheck all retained initial bytes and the separate regeneration report."""
    from initialization_parent import check_verification
    selected=[s for s in plan['stages'] if s['validation_binding'] is not None]
    record,verify=selected
    def retained(stage,root):
        from types import SimpleNamespace
        result=retained_stage(stage,output,profile)
        relative=remote_name(SimpleNamespace(profile=profile),root)
        candidates=[e for e in result['exports'] if relative==e['remote_root']]
        if len(candidates)!=1:raise EvidenceError('initialization selected output missing or ambiguous')
        return candidates[0]
    parent=retained(record,verify['parent_record_root'])
    template=read_json(output/'derived'/verify['name']/'job.json')
    argv=template['argv'];args=argv[argv.index('--')+1:]
    if args.count('--output')!=1:raise EvidenceError('one explicit regeneration output required')
    destination=args[args.index('--output')+1]
    if destination not in template['export_roots']:raise EvidenceError('regeneration output must be fully retained')
    result=retained(verify,destination)
    return check_verification(Path(result['directory']),result['files'],Path(parent['directory']),parent['files'],verify['parent_binding'])


def finalize(health,plan,expected,output,result,health_file,stop):
    fields(result,'schema plan_sha256 pod_id outcome stages unstarted_stages failure scope production_acceptance provider_termination','sustained result')
    if (result['schema']!='ovl.sustained-workload-result.v1' or result['plan_sha256']!=expected or result['pod_id']!=health.pod
        or result['production_acceptance']!='NOT_RUN' or result['provider_termination']!='SEPARATE_CONTROLLER_REQUIRED'):
        raise EvidenceError('retained sustained result identity differs')
    history=result['stages']
    profile=read_json(output/'selected-profile.json')
    if digest(profile)!=plan['profile_sha256']:raise EvidenceError('retained endpoint profile differs from selected plan')
    if type(history) is not list or not 1<=len(history)<=len(plan['stages']):raise EvidenceError('complete retained stage prefix required')
    for entry,stage in zip(history,plan['stages']):
        selection=read_json(output/'derived'/stage['name']/'selection.json');job=selection['job_sha256']
        actual=retained_stage(stage,output,profile);check_export_budget(stage,actual)
        if entry!={'name':stage['name'],'job_sha256':job,'result_sha256':digest(actual),'exit':actual['exit']}:
            raise EvidenceError('saved sustained result differs from retained bytes')
    if set(health.jobs)!={e['job_sha256'] for e in history}:raise EvidenceError('sustained result omits launched jobs')
    if result['unstarted_stages']!=[s['name'] for s in plan['stages'][len(history):]]:raise EvidenceError('unstarted sustained stages differ')
    failed=[i for i,e in enumerate(history) if e['exit']['state']!='EXITED' or e['exit']['exit_code']!=0]
    if result['failure'] is not None:
        failure=read_json(output/'stages'/history[-1]['name']/'dispatch-failure.json')
        if result['failure']!=failure or failure['job_sha256']!=history[-1]['job_sha256'] or result['outcome']!='DISPATCH_FAILED_AFTER_RETENTION':
            raise EvidenceError('saved dispatcher failure does not match retained stage')
        if failed and failed!=[len(history)-1]:raise EvidenceError('stage continued after a failed predecessor')
    elif failed:
        if failed!=[len(history)-1] or result['outcome']!='FAILED_OR_ABANDONED':raise EvidenceError('sustained failure outcome differs')
    elif result['outcome']=='STOPPED_AFTER_STAGE':
        if not stop.exists() and not journal_stop(stop.parent,plan['rental_intent_sha256'],health.pod):raise EvidenceError('retained sustained stop lacks original marker')
    elif result['outcome']!='EXITED_ZERO' or len(history)!=len(plan['stages']):raise EvidenceError('incomplete sustained work cannot report zero exits')
    if plan['schema']=='ovl.initialization-cycle-plan.v1' and result['outcome']=='EXITED_ZERO':
        save_once(output/'initialization-consistency.json',initialization_result(plan,output,profile))
    final=output/'final';names=['result.json']
    for i,entry in enumerate(history):
        dest=final/f'stage-{i:03d}';dest.mkdir(mode=0o700,exist_ok=True)
        name='exit.json' if entry['exit']['state']=='EXITED' else 'abandoned.json'
        save_once(dest/name,entry['exit']);names.append(f'stage-{i:03d}/{name}')
    if health.complete:
        files=read_json(health.journal.directory/'final-export-inventory.json');health._check_final(final,files)
        if health.lifetime.remaining()>0:health.write(health_file)
    else:health.finish(final,inventory(final,names));health.write(health_file)
    return result


def run(plan,expected,rental,controller_directory,watchdog_file,transport,inputs,worker,output,health_file,*,sleep=time.sleep):
    watchdog,rental_plan,stages=validate(plan,expected,rental,transport,inputs,worker)
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    stop=Path(controller_directory)/'stop-request.json'
    parent_kind='initialization' if plan['schema']=='ovl.initialization-cycle-plan.v1' else 'pilot'
    with Journal(output/'health-journal').lease() as journal:
        if journal.events and not(output/'selected-plan.json').exists():raise EvidenceError('sustained health lost its selected plan')
        save_once(output/'selected-plan.json',plan)
        save_once(output/'selected-profile.json',transport.profile)
        bindings={};download_bindings={};derived={}
        # Reconstruct every historical contract before health journal adoption,
        # including completed jobs. No peer-supplied replacement binding is used.
        for stage,template in stages:
            if not(output/'derived'/stage['name']/'selection.json').exists():continue
            parent=parent_for(stage,stages,output,transport)
            selection={k:stage[k] for k in ('name','template_sha256','work_seconds','export_reserve_seconds','parent_record_root')}
            value=derive(selection,template,output/'derived'/stage['name'],expected,rental_plan,0,parent=parent,parent_kind=parent_kind)
            derived[stage['name']]=value
            if stage['validation_binding'] is not None:bindings[value[1]]=stage['validation_binding']
            if stage['download_binding'] is not None:download_bindings[value[1]]=stage['download_binding']
        health_class=SustainedHealth
        if parent_kind=='initialization':
            from initialization_health import InitializationCycleHealth
            health_class=InitializationCycleHealth
        health=health_class(journal,watchdog,transport.profile['pod_id'],bindings,download_bindings)
        if health.complete or (output/'final/result.json').exists():
            return finalize(health,plan,expected,output,read_json(output/'final/result.json'),health_file,stop)
        upload_dir=output/'uploads';upload_dir.mkdir(mode=0o700,exist_ok=True)
        for u in plan['uploads']:
            identity=digest({'profile_sha256':plan['profile_sha256'],'upload':u});receipt=upload_dir/(identity+'.json')
            if receipt.exists():
                saved=read_json(receipt)
                if saved.get('selection')!=u or saved.get('operation_sha256')!=identity:raise EvidenceError('sustained upload receipt changed')
                continue
            if health.jobs:raise EvidenceError('launched stage lacks original input transfer receipt')
            retain_observation(output,controller_observation(controller_directory,rental,health.pod,starting=True))
            watchdog_heartbeat(watchdog_file,watchdog,health.now(),health.pod)
            if stop.exists() or health.now()>=rental_plan['request_checkpoint_epoch']:raise EvidenceError('stop forbids setup transfer')
            local=confined(Path(inputs),u['path'])
            if file_hash(local)!=u['sha256']:raise EvidenceError('selected input bytes changed')
            def progress(counts):health.bytes(identity,counts,total=u['bytes'],direction='send');health.write(health_file)
            seconds=(u['bytes']+plan['timing']['transfer_floor_bytes_per_second']-1)//plan['timing']['transfer_floor_bytes_per_second']+30
            result=transport.put(u['remote_path'],local,min(rental_plan['request_checkpoint_epoch'],health.now()+seconds),progress=progress)
            if result['sha256']!=u['sha256'] or result['bytes_sent']!=u['bytes']:raise EvidenceError('uploaded input differs')
            save_once(receipt,{'selection':u,'operation_sha256':identity,'transfer':result})
        history=[];outcome='EXITED_ZERO';failure=None
        for stage,template in stages:
            observation=controller_observation(controller_directory,rental,health.pod,starting=False)
            retain_observation(output,observation)
            started=stage['name'] in derived and derived[stage['name']][1] in health.jobs
            if observation['terminated']:raise EvidenceError('rental terminated before complete retained result')
            if not started and (observation['stopping'] or stop.exists()):
                if not history:raise EvidenceError('stop before any stage; guard owns termination')
                outcome='STOPPED_AFTER_STAGE';break
            if not started:watchdog_heartbeat(watchdog_file,watchdog,health.now(),health.pod)
            if stage['name'] not in derived:
                parent=parent_for(stage,stages,output,transport)
                selection={k:stage[k] for k in ('name','template_sha256','work_seconds','export_reserve_seconds','parent_record_root')}
                derived[stage['name']]=derive(selection,template,output/'derived'/stage['name'],expected,rental_plan,health.now(),parent=parent,parent_kind=parent_kind)
            path,job_root,job=derived[stage['name']]
            if stage['validation_binding'] is not None:bindings[job_root]=stage['validation_binding']
            if stage['download_binding'] is not None:download_bindings[job_root]=stage['download_binding']
            hook=None
            if stage['retention'] is not None:
                retention_class=CheckpointRetention if stage['retention']['schema']=='ovl.pilot-checkpoint-retention.v1' else InitialRetention
                hook=retention_class(transport,health,job,job_root,stage['retention'],output/'initial-retention'/stage['name'],health_file,output/'objects')
            elif not started and job['deadline_epoch']+stage['export_reserve_seconds']>health.exported+1800:
                raise EvidenceError('uncheckpointed stage cannot fit unchanged export age')
            stage_output=output/'stages'/stage['name'];failure_path=stage_output/'dispatch-failure.json'
            if failure_path.exists():
                failure=read_json(failure_path)
                if failure.get('job_sha256')!=job_root:raise EvidenceError('retained dispatcher failure changed job identity')
            if failure is None:
                try:
                    result=run_stage(transport,health,path,job_root,worker,plan['worker_sha256'],stage_output,
                                     health_file,stop,plan['rental_intent_sha256'],sleep=sleep,initial_retention=hook,
                                     **({} if type(hook) is not CheckpointRetention else {'terminal_limits':{'maximum_bytes':stage['maximum_export_bytes'],
                                         'maximum_uncached_bytes':hook.selection['maximum_uncached_export_bytes']}}))
                except Exception as error:
                    if not(stage_output/'launch/launch-intent.json').exists():raise
                    failure={'schema':'ovl.sustained-stage-dispatch-failure.v1','job_sha256':job_root,'error_type':type(error).__name__,
                             'scope':'dispatch failed; request owned stage stop and preserve complete terminal outputs'}
                    save_once(failure_path,failure)
            if failure is not None:
                from sustained_pilot_abort import stop_and_retain
                result=stop_and_retain(transport,health,path,job_root,worker,plan['worker_sha256'],stage_output,health_file,stop,
                                      plan['rental_intent_sha256'],sleep=sleep,initial_retention=hook,
                                      **({} if type(hook) is not CheckpointRetention else {'terminal_limits':{'maximum_bytes':stage['maximum_export_bytes'],
                                          'maximum_uncached_bytes':hook.selection['maximum_uncached_export_bytes']}}))
            checked=retained_stage(stage,output,transport.profile)
            if checked!=result:raise EvidenceError('retained sustained result changed')
            check_export_budget(stage,result)
            history.append({'name':stage['name'],'job_sha256':job_root,'result_sha256':digest(result),'exit':result['exit']})
            if failure is not None:outcome='DISPATCH_FAILED_AFTER_RETENTION';break
            if result['exit']['state']!='EXITED' or result['exit']['exit_code']!=0:outcome='FAILED_OR_ABANDONED';break
        final=output/'final';final.mkdir(mode=0o700,exist_ok=True)
        result={'schema':'ovl.sustained-workload-result.v1','plan_sha256':expected,'pod_id':health.pod,'outcome':outcome,'stages':history,
                'unstarted_stages':[s['name'] for s,_ in stages[len(history):]],'failure':failure,'scope':'complete selected stage bytes retained; numerical acceptance separate',
                'production_acceptance':'NOT_RUN','provider_termination':'SEPARATE_CONTROLLER_REQUIRED'}
        save_once(final/'result.json',result)
        return finalize(health,plan,expected,output,result,health_file,stop)


def main():
    import argparse
    from pod_transfer import Transport
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','rental-intent','controller-journal','watchdog-heartbeat','profile','key','known-hosts','inputs','worker','output','health'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);a=p.parse_args()
    try:
        transport=Transport(read_json(a.profile),a.key,a.known_hosts)
        result=run(read_json(a.plan),a.plan_sha256,read_json(a.rental_intent),a.controller_journal,a.watchdog_heartbeat,
                   transport,a.inputs,a.worker,a.output,a.health)
        print(result['schema']+' '+digest(result))
    except Exception as error:p.exit(1,'sustained dispatcher refused: '+type(error).__name__+'; preserve evidence and existing guards\n')


if __name__=='__main__':main()
