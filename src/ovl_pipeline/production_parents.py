"""Bind actual preparation, initialization and pilot reports to a registration.

These checks authenticate no signature and execute no training. The caller must
first select/verify the public registration and source anchors independently.
Report consistency is not evidence that an operator's assertions are truthful.
"""
from .canonical import EvidenceError,digest,require_digest
from .schema import integer,fields,control
from .production_contract import validate


def equal(actual,expected,label):
    if actual!=expected:raise EvidenceError('production parent mismatch: '+label)


INITIAL_RECORD_FIELDS='schema scope result recipe kernel warmup_updates warmup_weights_discarded stream_sha256 code_root environment checkpoint control parameter_count process_observation production_admission'
INITIAL_VERIFICATION_FIELDS='schema result record_sha256 initial_state_sha256 recipe_sha256 code_root stream_sha256 environment warmup_updates scope prover_tensors_loaded_as_state process_observation distinct_process_from_record process_identity_scope performed_by independent_third_party production_admission'


def initialization_report_shape(value, *, verification=False):
    """Check nested structural metadata without claiming tensor regeneration.

    The compatible runtime object is separately hash-bound to the selected host.
    Its technical descriptions still require semantic publication review; this
    validator is not a general-purpose privacy classifier.
    """
    e=value['environment']
    fields(e,'compatible reproducibility_validation production_admission','initial environment envelope')
    if type(e['compatible']) is not dict:raise EvidenceError('invalid compatible runtime')
    for key in ('reproducibility_validation','production_admission'):
        equal(e[key],'NOT_RUN','initial environment scope')
    integer(value['warmup_updates'],0,10000,'initial warmup updates')
    for key in ('code_root','stream_sha256'):require_digest(value[key])
    if verification:
        for key in ('record_sha256','recipe_sha256','initial_state_sha256'):require_digest(value[key])
        return
    from .schema import recipe
    recipe(value['recipe'],gpu=True)
    fields(value['kernel'],'schema precision','initial kernel')
    if value['kernel']['schema']!='ovl.gpu-kernel.v1' or value['kernel']['precision'] not in ('bf16','fp32'):
        raise EvidenceError('invalid initial kernel')
    control(value['control'])
    c=value['control']
    if c['phase']!='wikipedia' or any(c[k]!=0 for k in ('global_step','phase_step','cursor')):
        raise EvidenceError('initial state retained progress')
    integer(value['parameter_count'],1,2**40,'initial parameter count')
    checkpoint=value['checkpoint'];fields(checkpoint,'schema state_root files','initial checkpoint')
    if checkpoint['schema']!='ovl.checkpoint.v1':raise EvidenceError('invalid initial checkpoint schema')
    require_digest(checkpoint['state_root'])
    entries=checkpoint['files']
    if type(entries) is not list or len(entries)!=2:raise EvidenceError('incomplete initial checkpoint inventory')
    for entry in entries:
        fields(entry,'path bytes sha256','initial checkpoint file')
        integer(entry['bytes'],1,2*1024**3,'initial checkpoint bytes');require_digest(entry['sha256'])
    if [x['path'] for x in entries]!=['state.json','state.safetensors']:
        raise EvidenceError('invalid initial checkpoint inventory')


def reject_private_process_fields(value):
    """Reject OS process observations anywhere in a new public report packet.

    This narrowly prevents the observed nested-field leak. It does not replace
    exact semantic review of every emitted artifact or the publication scanner.
    """
    if type(value) is dict:
        if {'process_observation','pid','boot_id','start_ticks'} & set(value):
            raise EvidenceError('private process field in public report')
        for child in value.values():reject_private_process_fields(child)
    elif type(value) is list:
        for child in value:reject_private_process_fields(child)


def private_process_commitment(p):
    import re
    fields(p,'pid boot_id start_ticks','private process observation')
    integer(p['pid'],1,2**31-1,'private process PID')
    if (type(p['boot_id']) is not str or not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',p['boot_id'])
            or type(p['start_ticks']) is not str or not re.fullmatch(r'(?:0|[1-9][0-9]{0,23})',p['start_ticks'])):
        raise EvidenceError('noncanonical private process observation')
    return digest({'schema':'ovl.private-process-observation.v1','observation':p})


def public_initialization(initial):
    """Project checked local reports without disclosing OS process identities.

    Commitments preserve the process-inequality assertion, not an attestation.
    Original reports remain local; all numerical parents and state roots survive.
    The derivative has its own schema and freshly recomputed report-parent hash.
    """
    from copy import deepcopy
    record=initial['record'];verified=initial['verification']
    fields(record,INITIAL_RECORD_FIELDS,'local initialization record')
    fields(verified,INITIAL_VERIFICATION_FIELDS,'local initialization verification')
    equal(record['schema'],'ovl.initialization-record.v1','local initial schema')
    equal(verified['schema'],'ovl.initialization-verification.v1','local regeneration schema')
    equal(verified['record_sha256'],digest(record),'local regeneration parent')
    equal(verified['process_identity_scope'],'operator OS observation, not hardware attestation','local process scope')
    observations=[]
    for value in (record,verified):
        initialization_report_shape(value,verification=value is verified)
        observations.append(private_process_commitment(value['process_observation']))
    if observations[0]==observations[1]:raise EvidenceError('same initialization process')
    public=[]
    for value,commitment,kind in zip((record,verified),observations,('record','verification')):
        value=deepcopy(value);del value['process_observation']
        value['schema']='ovl.public-initialization-'+kind+'.v1'
        value['process_observation_commitment']=commitment;public.append(value)
    public[1]['record_sha256']=digest(public[0])
    public[1]['process_identity_scope']='operator process commitment, not hardware attestation'
    result={'record':public[0],'verification':public[1]}
    reject_private_process_fields(result)
    return result


def validate_parents(registration,*,source,source_policy,prepared,initial_record,initial_verification,pilot_records,pilot_replays):
    contract=validate(registration);r=registration
    equal(digest(source),r['source_statement_sha256'],'source statement')
    equal(source.get('schema'),'ovl.source-preparation.v2','source schema')
    equal(source.get('scope'),'production-source-preparation','source scope')
    equal(source.get('run_id'),r['run_id'],'source run')
    equal(digest(source_policy),r['source_policy_sha256'],'source policy')
    equal(source_policy.get('statement_sha256'),digest(source),'source policy statement')
    equal(source_policy.get('source_revision'),source.get('source_revision'),'source signing revision')
    equal(digest(prepared),r['preparation_sha256'],'prepared manifest')
    equal(prepared.get('schema'),'ovl.complete-preparation.v1','preparation schema')
    equal(prepared.get('source_commitment_sha256'),digest(source),'preparation source')
    equal(prepared.get('code'),source.get('code'),'preparation code')
    equal(prepared.get('environment'),source.get('environment'),'preparation environment')
    if prepared.get('validation_used_for_training') is not False:raise EvidenceError('official validation must be held out')
    equal(set(prepared['streams']),{'wikipedia','conversation','conversation-validation'},'complete prepared splits')
    if digest(prepared['streams']['conversation'])==digest(prepared['streams']['conversation-validation']):
        raise EvidenceError('training and held-out validation manifests are identical')
    equal(prepared['tokenizer']['vocab_size'],r['recipe']['model']['vocab_size'],'actual tokenizer vocabulary')
    equal(prepared['conversation_selection']['source_inventory_root'],digest(source['conversation']['inventory']),'conversation public source')
    equal(prepared['conversation_selection']['split_filenames'],source['conversation']['splits'],'conversation official splits')
    equal(digest(initial_verification),r['initialization']['regeneration_report_sha256'],'initial regeneration report')
    public=initial_record.get('schema')=='ovl.public-initialization-record.v1'
    if public:
        reject_private_process_fields([registration,source,source_policy,prepared,initial_record,initial_verification,pilot_records,pilot_replays])
        fields(initial_record,INITIAL_RECORD_FIELDS.replace('process_observation','process_observation_commitment'),'public initial record')
        fields(initial_verification,INITIAL_VERIFICATION_FIELDS.replace('process_observation','process_observation_commitment'),'public initial verification')
        initialization_report_shape(initial_record)
        initialization_report_shape(initial_verification,verification=True)
        for value in (initial_record,initial_verification):require_digest(value['process_observation_commitment'])
        equal(initial_verification['process_identity_scope'],'operator process commitment, not hardware attestation','public process scope')
        equal(initial_verification['performed_by'],'project-operator','public verifier scope')
        if initial_verification['independent_third_party'] is not False:raise EvidenceError('public process commitment is not independent verification')
        for value in (initial_record,initial_verification):equal(value['production_admission'],'NOT_RUN','public initial admission')
    equal(initial_verification.get('schema'),'ovl.public-initialization-verification.v1' if public else 'ovl.initialization-verification.v1','initial report schema')
    equal(initial_verification.get('result'),'PASS','initial regeneration result')
    equal(initial_verification.get('scope'),'complete-initial-state-regenerated-and-compared','initial verification scope')
    if (initial_verification.get('prover_tensors_loaded_as_state') is not False
            or initial_verification.get('distinct_process_from_record') is not True):raise EvidenceError('initial state was not freshly regenerated')
    equal(initial_verification['record_sha256'],digest(initial_record),'initial record')
    equal(initial_record.get('schema'),'ovl.public-initialization-record.v1' if public else 'ovl.initialization-record.v1','initial record schema')
    equal(initial_record.get('scope'),'preproduction-regenerated-initial-state','initial record scope')
    equal(initial_record.get('result'),'RECORDED_AWAITING_FRESH_REGENERATION','initial record result')
    if initial_record.get('warmup_weights_discarded') is not True:raise EvidenceError('initial record retained warmup weights')
    equal(initial_record['recipe'],r['recipe'],'initial recipe')
    equal(initial_record['kernel'],r['kernel'],'initial precision')
    equal(initial_record['code_root'],r['code_root'],'initial code')
    equal(initial_verification['code_root'],r['code_root'],'initial verifier code')
    equal(initial_verification['recipe_sha256'],digest(r['recipe']),'initial verifier recipe')
    equal(initial_verification['initial_state_sha256'],r['initialization']['state_sha256'],'regenerated initial state')
    equal(initial_record['checkpoint']['state_root'],r['initialization']['state_sha256'],'committed initial state')
    equal(initial_record['warmup_updates'],r['initialization']['warmup_updates'],'record warmup')
    equal(initial_verification['warmup_updates'],r['initialization']['warmup_updates'],'verifier warmup')
    equal(initial_record['stream_sha256'],digest(prepared['streams']['wikipedia']),'initial warmup source')
    equal(initial_verification['stream_sha256'],digest(prepared['streams']['wikipedia']),'verified warmup source')
    process_key='process_observation_commitment' if public else 'process_observation'
    if initial_record[process_key]==initial_verification[process_key]:raise EvidenceError('same initialization process')
    for e in (initial_record['environment'],initial_verification['environment']):
        equal(e['compatible'].get('schema'),'ovl.gpu-environment.v1','GPU environment profile')
        equal(e['compatible'].get('kernel'),r['kernel'],'GPU environment kernel')
        equal(digest(e['compatible']),r['runtime']['compatible_environment_sha256'],'initial numerical environment')
    equal(set(pilot_records),{'wikipedia','conversation'},'both pilot records')
    equal(set(pilot_replays),{'wikipedia','conversation'},'both pilot replays')
    for phase in ('wikipedia','conversation'):
        census=r['coverage'][phase];record=pilot_records[phase];replay=pilot_replays[phase]
        equal(census['stream_sha256'],digest(prepared['streams'][phase]),'census source')
        equal(census['documents'],prepared['streams'][phase]['documents'],'census document count')
        equal(census['targets'],prepared['streams'][phase]['targets'],'census target count')
        equal(digest(record),r['pilots'][phase]['record_sha256'],'pilot record')
        equal(digest(replay),r['pilots'][phase]['replay_sha256'],'pilot replay')
        equal(record.get('schema'),'ovl.gpu-pilot-record.v1','pilot schema')
        equal(record.get('scope'),'development-gpu-pilot-only','pilot scope')
        if replay.get('schema') not in ('ovl.gpu-pilot-replay.v1','ovl.gpu-pilot-replay.v2'):raise EvidenceError('unknown replay schema')
        equal(replay.get('scope'),'fresh-initialization-continuous-pilot-replay','full pilot replay scope')
        equal(replay.get('result'),'PASS','pilot replay result')
        if replay.get('resume_from') is not None or replay.get('initial_state_regenerated') is not True:
            raise EvidenceError('segment/resume pilot cannot replace complete replay')
        equal(replay['record_sha256'],digest(record),'replay record parent')
        settings=record['settings']
        from .pilot_delivery import settings_delivery, policy as delivery_policy
        delivered=settings_delivery(settings)
        if delivered is not None and replay['schema']!='ovl.gpu-pilot-replay.v2':raise EvidenceError('pilot replay omits delivery')
        if replay['schema']=='ovl.gpu-pilot-replay.v2':
            rp=delivery_policy(replay['delivery'])
            if rp['mode']!='replay' or rp['phase']!=phase:raise EvidenceError('pilot replay delivery differs')
        equal(settings.get('scope'),'development-gpu-pilot-only','pilot settings scope')
        if settings.get('requested_updates') is not None:raise EvidenceError('fixed-update pilot cannot admit sustained forecast')
        integer(settings.get('requested_seconds'),600,3600,'sustained pilot duration')
        equal(settings['recipe'],r['recipe'],'pilot recipe')
        equal(settings['kernel'],r['kernel'],'pilot kernel')
        equal(settings['stream'],prepared['streams'][phase],'pilot full-stream parent')
        equal(settings['code_root'],r['code_root'],'pilot code')
        equal(settings['warmup_updates'],r['initialization']['warmup_updates'],'pilot warmup')
        for e in (settings['environment'],replay['environment']):
            equal(digest(e['compatible']),r['runtime']['compatible_environment_sha256'],'pilot numerical environment')
        equal(replay['updates_recomputed'],record['updates'],'complete pilot updates')
        integer(record['updates'],1,1_000_000,'recorded pilot updates')
        interval=settings['checkpoint_every'];integer(interval,1,1_000_000,'pilot checkpoint interval')
        if (record['updates']+interval-1)//interval+1>4096:raise EvidenceError('pilot chain exceeds bound')
        steps=list(range(0,record['updates']+1,interval))
        if steps[-1]!=record['updates']:steps.append(record['updates'])
        equal(len(record['boundaries']),len(steps),'complete pilot boundary schedule')
        equal(record['timed_checkpoints'],len(steps)-1,'timed pilot checkpoints')
        equal(len(replay['compared']),len(record['boundaries']),'all pilot boundaries')
        if replay.get('verifier_checkpoint_overhead_included') is not True:
            raise EvidenceError('pilot replay omits verifier checkpoint overhead')
        equal(replay.get('verifier_checkpoints_saved'),len(record['boundaries']),'complete verifier checkpoint count')
        previous=digest(settings)
        for index,(expected,actual) in enumerate(zip(record['boundaries'],replay['compared'])):
            fields(expected,'index step control path checkpoint previous','pilot boundary')
            fields(actual,'index state_root','pilot comparison')
            equal(expected['index'],index,'pilot sequential boundary index')
            equal(expected['step'],steps[index],'pilot scheduled step')
            c=expected['control'].copy();cycle=c.pop('pilot_cycle',None);control(c)
            integer(cycle,0,record['updates'],'pilot cycle')
            if c['phase_step']!=steps[index] or c['cursor']>settings['stream']['targets']:
                raise EvidenceError('pilot control differs from full stream/schedule')
            if index==0 and (c['cursor']!=0 or cycle!=0):raise EvidenceError('pilot did not start at stream origin')
            if index and cycle*settings['stream']['targets']+c['cursor']<=previous_targets:
                raise EvidenceError('pilot target progress regressed')
            previous_targets=cycle*settings['stream']['targets']+c['cursor']
            equal(expected['control']['global_step'],steps[index],'pilot control step')
            equal(expected['control']['phase'],phase,'pilot control phase')
            equal(expected['path'],f'boundary-{index:05d}','pilot checkpoint path')
            equal(expected['previous'],previous,'pilot boundary ancestry')
            equal(actual['index'],expected['index'],'pilot boundary index')
            equal(actual['state_root'],expected['checkpoint']['state_root'],'pilot boundary state')
            previous=digest(expected)
        equal(previous_targets,record['measured_targets'],'pilot complete measured target accounting')
        f=r['forecast_input']['phases'][phase]
        mapping={'measured_full_batch_updates':'measured_full_batch_updates','measured_ms':'measured_ms',
                 'eligible_duration_for_forecast':'eligible_duration_for_forecast','measured_updates':'updates',
                 'measured_checkpoints':'timed_checkpoints','warmup_excluded':'warmup_excluded','overhead_included':'overhead_included'}
        for fk,rk in mapping.items():equal(f[fk],record[rk],'forecast measured '+fk)
        equal(f['measured_checkpoint_every'],settings['checkpoint_every'],'forecast checkpoint interval')
        equal(f['replay_measured_ms'],replay['measured_ms'],'forecast replay time')
        if replay.get('eligible_for_forecast_comparison') is not True:raise EvidenceError('ineligible forecast replay')
        for name in ('measured_targets','measured_full_batch_updates','timed_checkpoints'):
            equal(replay[name],record[name],'replayed measured work '+name)
    return {'schema':'ovl.production-parent-check.v1','result':'PASS','registration_sha256':digest(r),
            'scope':'report-content-and-parent-consistency-only','contract':contract,
            'publisher_identity':'NOT_RUN','raw_reconstruction':'NOT_RUN','training_replay':'NOT_RUN',
            'assertion_truth_established':False,'production_admission':'NOT_RUN'}
