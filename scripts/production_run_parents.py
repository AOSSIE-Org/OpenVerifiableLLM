"""Select production parents from complete retained same-rental work.

No report-only shortcut: all required pilot states and original job exits are
checked before their reports enter a registration packet. These are operator
execution observations, not an independent witness or production admission.
"""
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.schema import fields
from pilot_record_parent import check as check_record
from verify_pilot_cycle import replay_states,verify as check_cycle
from run_sustained_pilot import retained_stage,initialization_result


def qualification(plan,expected,output,profile,*,prepared_qualification=None):
    if digest(plan)!=expected:raise EvidenceError('qualification plan differs from original selection')
    wanted={'setup','prepared-inputs','cuda-record','cuda-replay','cuda-resume',
            'wikipedia-record','wikipedia-replay','conversation-record','conversation-replay'}
    if prepared_qualification is not None:
        if (prepared_qualification.get('result')!='PASS' or prepared_qualification['pod_id']!=profile['pod_id']
            or 'prepared_qualification_sha256' in prepared_qualification):
            raise EvidenceError('only one same-host optimization may reuse checked preparation')
        wanted-={'setup','prepared-inputs'}
    stages={s['name']:s for s in plan['stages']}
    if len(stages)!=len(plan['stages']) or set(stages)!=wanted:
        raise EvidenceError('complete focused qualification cycle and both sustained pairs required')
    final=read_json(output/'final/result.json')
    if (final['plan_sha256']!=expected or final['pod_id']!=profile['pod_id'] or final['outcome']!='EXITED_ZERO'
        or final['unstarted_stages']!=[] or final['failure'] is not None
        or [e['name'] for e in final['stages']]!=[s['name'] for s in plan['stages']]):
        raise EvidenceError('qualification did not finish its entire original phase')
    retained={}
    for entry in final['stages']:
        name=entry['name'];result=retained_stage(stages[name],output,profile)
        if (entry['job_sha256']!=result['job_sha256'] or entry['result_sha256']!=digest(result)
            or entry['exit']!=result['exit'] or result['exit']['state']!='EXITED' or result['exit']['exit_code']!=0):
            raise EvidenceError('qualification lacks actual successful retained exits')
        retained[name]=result
    def numerical(name):
        stage=stages[name];root=stage['retention']['output_root']
        relative=root.removeprefix(profile['remote_root']+'/')
        selected=[e for e in retained[name]['exports'] if e['remote_root']==relative]
        if len(selected)!=1:raise EvidenceError('qualification numerical output differs from selection')
        return selected[0]
    tiny={mode:numerical('cuda-'+mode) for mode in ('record','replay','resume')}
    record=read_json(Path(tiny['record']['directory'])/'record.json')
    cycle=check_cycle(tiny['record']['directory'],tiny['record']['files'],stages['cuda-record']['parent_binding'],digest(record),
        tiny['replay']['directory'],tiny['replay']['files'],tiny['resume']['directory'],tiny['resume']['files'],resume_from=1)
    reports={};replays={};checks={};settings=[record['settings']];profiles={}
    for phase in ('wikipedia','conversation'):
        recording=numerical(phase+'-record');replaying=numerical(phase+'-replay')
        parent=check_record(Path(recording['directory']),recording['files'],stages[phase+'-record']['parent_binding'])
        checks[phase]=replay_states(Path(replaying['directory']),replaying['files'],parent['record'],parent['record_sha256'],resume_from=None)
        reports[phase]=parent['record'];replays[phase]=read_json(Path(replaying['directory'])/'verification.json')
        settings.append(parent['record']['settings'])
        for mode,entry,report in [('record',recording,reports[phase]),('replay',replaying,replays[phase])]:
            path=Path(entry['directory'])/'timing.json'
            if path.exists():
                from ovl_pipeline.phase_timing import validate
                profiles[phase+'-'+mode]=validate(read_json(path),report,scope='operator-pilot-phase-timing')
    common=settings[0]
    for selected in settings[1:]:
        if any(selected[k]!=common[k] for k in ('recipe','kernel','code_root','warmup_updates')):
            raise EvidenceError('qualification stages use different numerical selections')
        if selected['environment']['compatible']!=common['environment']['compatible']:
            raise EvidenceError('qualification stages do not share a compatible host runtime')
    result={'schema':'ovl.retained-same-host-qualification.v1','result':'PASS','plan_sha256':expected,
        'pod_id':profile['pod_id'],'cycle':cycle,'pilot_records':reports,'pilot_replays':replays,'pair_checks':checks,
        'compatible_environment_sha256':digest(common['environment']['compatible']),
        'scope':'complete retained states and successful job observations; no new numerical execution or production admission',
        'production_admission':'NOT_RUN','independent_third_party':False}
    if profiles:result['profiles']=profiles
    if prepared_qualification is not None:
        for phase in ('wikipedia','conversation'):
            before=prepared_qualification['pilot_records'][phase]['settings'];after=reports[phase]['settings']
            if any(before[k]!=after[k] for k in ('code_root','stream','kernel','warmup_updates')):
                raise EvidenceError('optimization must reuse the same frozen code and complete prepared inputs')
        if result['compatible_environment_sha256']!=prepared_qualification['compatible_environment_sha256']:
            raise EvidenceError('optimization host runtime changed')
        result['prepared_qualification_sha256']=digest(prepared_qualification)
    return result


def initialization(plan,expected,output,profile,qualified):
    if digest(plan)!=expected:raise EvidenceError('initialization plan differs from original selection')
    result=read_json(output/'final/result.json')
    if (result['plan_sha256']!=expected or result['pod_id']!=qualified['pod_id'] or result['pod_id']!=profile['pod_id']
        or result['outcome']!='EXITED_ZERO' or result['unstarted_stages'] or result['failure'] is not None
        or [e['name'] for e in result['stages']]!=[s['name'] for s in plan['stages']]):
        raise EvidenceError('initialization did not finish on the selected rental')
    checked=initialization_result(plan,output,profile)
    selected={}
    for stage in plan['stages']:
        action=stage['validation_binding']['action'];retained=retained_stage(stage,output,profile)
        entry=next(e for e in result['stages'] if e['name']==stage['name'])
        if entry['result_sha256']!=digest(retained) or retained['exit']['state']!='EXITED' or retained['exit']['exit_code']!=0:
            raise EvidenceError('initialization retained exit differs')
        # Select the numerical output from the original job invocation.
        job=read_json(output/'derived'/stage['name']/'job.json')
        args=job['argv'][job['argv'].index('--')+1:]
        if args.count('--output')!=1:raise EvidenceError('ambiguous initialization output')
        destination=args[args.index('--output')+1]
        if destination not in job['export_roots']:raise EvidenceError('initialization output was not fully retained')
        root=destination.removeprefix(profile['remote_root']+'/')
        exports=[e for e in retained['exports'] if e['remote_root']==root]
        if len(exports)!=1:raise EvidenceError('initialization output selection differs')
        name='record.json' if action=='record' else 'verification.json'
        selected[action]=read_json(Path(exports[0]['directory'])/name)
        if digest(selected[action]['environment']['compatible'])!=qualified['compatible_environment_sha256']:
            raise EvidenceError('initialization runtime differs from qualified host')
    if set(selected)!={'record','verify'}:raise EvidenceError('complete record and fresh initialization regeneration required')
    return {'record':selected['record'],'verification':selected['verify'],'retained_check':checked}


def packet(registration,source,source_bundle,source_policy,prepared,qualified,initial,output):
    """Write only the exact closed packet after validating every report link."""
    from ovl_pipeline.production_parents import validate_parents,public_initialization
    from ovl_pipeline.production_anchoring import PACKET_FILES
    from ovl_pipeline.canonical import file_hash
    initial=public_initialization(initial)
    parents={'source':source,'source_policy':source_policy,'prepared':prepared,'initial_record':initial['record'],
             'initial_verification':initial['verification'],'pilot_records':qualified['pilot_records'],
             'pilot_replays':qualified['pilot_replays']}
    checked=validate_parents(registration,**parents)
    if file_hash(source_bundle)!=registration['source_bundle_sha256']:raise EvidenceError('selected source bundle differs')
    if output.exists():raise EvidenceError('production packet must be fresh')
    values={'registration.json':registration,'source-statement.json':source,'preparation.json':prepared,
            'initial-record.json':initial['record'],'initial-verification.json':initial['verification']}
    for phase in ('wikipedia','conversation'):
        values[phase+'-pilot-record.json']=qualified['pilot_records'][phase]
        values[phase+'-pilot-replay.json']=qualified['pilot_replays'][phase]
    if set(values)|{'source-statement.sigstore.json'}!=PACKET_FILES:raise EvidenceError('incomplete registration packet')
    output.mkdir(parents=True)
    for name,value in values.items():write_json(output/name,value)
    (output/'source-statement.sigstore.json').write_bytes(source_bundle.read_bytes())
    return checked
