"""Bind actual preparation, initialization and pilot reports to a registration.

These checks authenticate no signature and execute no training. The caller must
first select/verify the public registration and source anchors independently.
Report consistency is not evidence that an operator's assertions are truthful.
"""
from .canonical import EvidenceError,digest
from .schema import integer,fields,control
from .production_contract import validate


def equal(actual,expected,label):
    if actual!=expected:raise EvidenceError('production parent mismatch: '+label)


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
    equal(initial_verification.get('schema'),'ovl.initialization-verification.v1','initial report schema')
    equal(initial_verification.get('result'),'PASS','initial regeneration result')
    equal(initial_verification.get('scope'),'complete-initial-state-regenerated-and-compared','initial verification scope')
    if (initial_verification.get('prover_tensors_loaded_as_state') is not False
            or initial_verification.get('distinct_process_from_record') is not True):raise EvidenceError('initial state was not freshly regenerated')
    equal(initial_verification['record_sha256'],digest(initial_record),'initial record')
    equal(initial_record.get('schema'),'ovl.initialization-record.v1','initial record schema')
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
    if initial_record['process_observation']==initial_verification['process_observation']:raise EvidenceError('same initialization process')
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
        equal(replay.get('schema'),'ovl.gpu-pilot-replay.v1','replay schema')
        equal(replay.get('scope'),'fresh-initialization-continuous-pilot-replay','full pilot replay scope')
        equal(replay.get('result'),'PASS','pilot replay result')
        if replay.get('resume_from') is not None or replay.get('initial_state_regenerated') is not True:
            raise EvidenceError('segment/resume pilot cannot replace complete replay')
        equal(replay['record_sha256'],digest(record),'replay record parent')
        settings=record['settings']
        equal(settings.get('schema'),'ovl.gpu-pilot-settings.v1','pilot settings schema')
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
