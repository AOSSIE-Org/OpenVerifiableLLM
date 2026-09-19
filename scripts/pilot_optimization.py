"""One bounded batch-size trial; estimates are distinct from measured timings.

Selection requires complete checked pilot parents from the coordinator. These
calculations confer no numerical/replay/production acceptance by themselves.
"""
from decimal import Decimal,ROUND_CEILING
from ovl_pipeline.canonical import EvidenceError,digest
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.phase_timing import validate as timing


def policy(value):
    fields(value,'schema minimum_gain_percent minimum_saved_seconds payback_multiplier coverage','single optimization policy')
    if value['schema']!='ovl.single-batch-optimization.v1':raise EvidenceError('unsupported single optimization')
    integer(value['minimum_gain_percent'],1,50,'minimum useful improvement')
    integer(value['minimum_saved_seconds'],1,86400,'minimum useful saving')
    integer(value['payback_multiplier'],1,10,'qualification payback')
    fields(value['coverage'],'baseline candidate','selected full coverage counts')
    for variant,phases in value['coverage'].items():
        fields(phases,'wikipedia conversation','complete phase coverage')
        for phase,census in phases.items():
            if census['phase']!=phase or census['schema']!='ovl.complete-schedule-counts.v1':raise EvidenceError('complete census required')
            for k in ('updates','targets','target_bearing_windows'):integer(census[k],1,2**53-1,k)
            if census['updates']!=(census['target_bearing_windows']+census['batch_size']-1)//census['batch_size']:
                raise EvidenceError('census batch arithmetic differs')
    for phase in ('wikipedia','conversation'):
        a=value['coverage']['baseline'][phase];b=value['coverage']['candidate'][phase]
        if any(a[k]!=b[k] for k in ('stream_sha256','targets','target_bearing_windows','context','documents','padded_positions','masked_context_positions')):
            raise EvidenceError('optimization changes corpus or context')
        if b['batch_size']!=2*a['batch_size']:raise EvidenceError('one batch-doubling candidate required')
    return value


def projection(q,counts):
    total=setup=Decimal(0)
    for phase,census in counts.items():
        record=q['pilot_records'][phase];replay=q['pilot_replays'][phase];settings=record['settings']
        if (digest(settings['recipe'])!=census['recipe_sha256'] or digest(settings['stream'])!=census['stream_sha256']
            or record['eligible_duration_for_forecast'] is not True or replay['eligible_for_forecast_comparison'] is not True
            or replay['record_sha256']!=digest(record) or replay['updates_recomputed']!=record['updates']):
            raise EvidenceError('complete matching sustained record/full replay required')
        for report in (record,replay):
            full=report['measured_full_batch_updates'];integer(full,1,1_000_000,'measured full batches')
            total+=Decimal(census['updates'])*Decimal(report['measured_ms'])/full
            setup+=report['setup_including_warmup_ms']
    return total+setup


def screen(q,value):
    policy(value);projection(q,value['coverage']['baseline'])
    totals={}
    for phase,census in value['coverage']['baseline'].items():
        for mode,report in [('record',q['pilot_records'][phase]),('replay',q['pilot_replays'][phase])]:
            profile=q.get('profiles',{}).get(phase+'-'+mode)
            if profile is None:raise EvidenceError('separate phase timings required before selecting optimization')
            timing(profile,report,scope='operator-pilot-phase-timing')
            entries={e['operation']:e for e in profile['measurements'] if e['phase']=='measured'}
            if entries.get('numerical_update',{}).get('calls')!=report.get('updates',report.get('updates_recomputed')):
                raise EvidenceError('timing must observe every measured numerical update')
            if entries['numerical_update']['cuda_calls']!=entries['numerical_update']['calls']:
                raise EvidenceError('CUDA timing must cover every measured numerical update')
            observed=sum(e['wall_ns'] for e in entries.values())
            wall=report['measured_ms']*1_000_000
            if observed>wall:raise EvidenceError('timing wall categories exceed enclosing elapsed time')
            scale=Decimal(census['updates'])/report['measured_full_batch_updates']
            for name,entry in entries.items():totals[name]=totals.get(name,Decimal(0))+entry['wall_ns']*scale
            totals['uninstrumented']=totals.get('uninstrumented',Decimal(0))+(wall-observed)*scale
    largest=max(totals,key=totals.get)
    numerical=totals.get('numerical_update',Decimal(0));all_time=sum(totals.values())
    try_candidate=largest=='numerical_update' and numerical*2>=all_time
    return {'schema':'ovl.single-optimization-screen.v1','policy_sha256':digest(value),'baseline_sha256':digest(q),
        'largest_projected_wall_category':largest,'projected_wall_ns':{k:int(v.to_integral_value(rounding=ROUND_CEILING)) for k,v in sorted(totals.items())},
        'decision':'TRY_ONE_BATCH_DOUBLING' if try_candidate else 'KEEP_BASELINE',
        'reason':'Numerical update is largest and at least half of projected elapsed work; measure one larger batch.' if try_candidate else
                 'Batch doubling is not justified by the demonstrated bottleneck; no speculative recipe change retained.',
        'scope':'operator timing extrapolation; no controlled 4090/5090 comparison, margins, or verification credit'}


def choose(baseline,candidate,value,qualification_ms):
    policy(value);integer(qualification_ms,1,7*86400*1000,'actual candidate qualification duration')
    if screen(baseline,value)['decision']!='TRY_ONE_BATCH_DOUBLING':raise EvidenceError('candidate not selected by observed bottleneck')
    if candidate.get('prepared_qualification_sha256')!=digest(baseline):raise EvidenceError('candidate must inherit exact checked baseline')
    for phase in ('wikipedia','conversation'):
        a=baseline['pilot_records'][phase]['settings']['recipe'];b=candidate['pilot_records'][phase]['settings']['recipe']
        if {k:v for k,v in a.items() if k not in ('batch_size','boundary_every')}!={k:v for k,v in b.items() if k not in ('batch_size','boundary_every')}:
            raise EvidenceError('unselected candidate numerical change')
        if b['batch_size']!=2*a['batch_size'] or b['boundary_every']*2!=a['boundary_every']:
            raise EvidenceError('candidate must preserve primary checkpoint token cadence')
        old_interval=baseline['pilot_records'][phase]['settings']['checkpoint_every']
        new_interval=candidate['pilot_records'][phase]['settings']['checkpoint_every']
        if new_interval!=(old_interval+1)//2:raise EvidenceError('candidate pilot must preserve checkpoint cadence per batch row')
    old=projection(baseline,value['coverage']['baseline']);new=projection(candidate,value['coverage']['candidate']);saved=old-new
    sufficient=(saved*100>=old*value['minimum_gain_percent'] and saved>=value['minimum_saved_seconds']*1000
                and saved>=qualification_ms*value['payback_multiplier'])
    return {'schema':'ovl.single-optimization-decision.v1','policy_sha256':digest(value),'baseline_sha256':digest(baseline),
        'candidate_sha256':digest(candidate),'selected':'candidate' if sufficient else 'baseline',
        'baseline_projected_training_replay_ms':int(old.to_integral_value(rounding=ROUND_CEILING)),
        'candidate_projected_training_replay_ms':int(new.to_integral_value(rounding=ROUND_CEILING)),
        'actual_candidate_qualification_ms':qualification_ms,
        'reason':'All declared gain and qualification-payback thresholds met.' if sufficient else 'Insufficient measured benefit; keep checked baseline.',
        'scope':'projection from actual same-host record/full-replay rates and complete corpus counts, no runtime or price margins; not actual production time or billing'}
