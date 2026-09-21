"""Deterministic production inputs from retained same-rental qualification.

This file does not provision, upload or launch. The coordinator separately checks
actual parents, public precommitment, original budget and every worker fence.
"""
from copy import deepcopy
from decimal import Decimal,ROUND_CEILING
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json
from ovl_pipeline.production_contract import checkpoint_count
from ovl_pipeline.budget import forecast
from sustained_pilot_selection import DEADLINE


def dollars(value):return format(value.quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')


def registration(template,qualified,initial,rental,now,*,construction_seconds=0):
    r=deepcopy(template);plan=rental['watchdog_intent']['plan'];inp=plan['input']
    if not inp['now_epoch']<=now<plan['request_checkpoint_epoch']:raise EvidenceError('registration outside original rental work window')
    r['runtime']['compatible_environment_sha256']=qualified['compatible_environment_sha256']
    r['initialization']={'state_sha256':initial['record']['checkpoint']['state_root'],
                        'regeneration_report_sha256':digest(initial['verification']),
                        'warmup_updates':initial['record']['warmup_updates']}
    from ovl_pipeline.schema import integer
    integer(construction_seconds,0,1500,'frozen registration construction exposure')
    if now+construction_seconds>=plan['request_checkpoint_epoch']:
        raise EvidenceError('registration construction exceeds original work window')
    rate=Decimal(inp['hourly_upper_usd']);elapsed=Decimal(now-inp['now_epoch'])*rate/3600
    # Move a bounded prospective construction allowance from fixed remainder
    # into committed exposure. Total lifetime reservation does not increase.
    committed_elapsed=elapsed+Decimal(construction_seconds)*rate/3600
    r['pilots']={};phases={}
    for phase,census in r['coverage'].items():
        recorded=qualified['pilot_records'][phase];replayed=qualified['pilot_replays'][phase]
        if recorded['settings']['recipe']!=r['recipe'] or recorded['settings']['code_root']!=r['code_root']:
            raise EvidenceError('registration template differs from selected numerical configuration')
        r['pilots'][phase]={'record_sha256':digest(recorded),'replay_sha256':digest(replayed)}
        phases[phase]={'updates':census['updates'],'training_completed':0,'replay_completed':0,
          'measured_full_batch_updates':recorded['measured_full_batch_updates'],'measured_ms':recorded['measured_ms'],
          'warmup_excluded':recorded['warmup_excluded'],'overhead_included':recorded['overhead_included'],
          'measurement_sha256':digest(recorded),'recipe_sha256':digest(r['recipe']),
          'stream_sha256':census['stream_sha256'],'schedule_sha256':digest(census),'replay_sha256':digest(replayed),
          'eligible_duration_for_forecast':recorded['eligible_duration_for_forecast'],
          'measured_updates':recorded['updates'],'replay_measured_ms':replayed['measured_ms'],
          'measured_checkpoints':recorded['timed_checkpoints'],'measured_checkpoint_every':recorded['settings']['checkpoint_every'],
          'production_checkpoint_every':r['recovery_every'],
          'production_checkpoints':checkpoint_count(census['updates'],r['recipe']['boundary_every'],r['recovery_every'])}
    base={'schema':'ovl.cost-forecast-input.v3','spent_usd':inp['spent_usd'],
          'committed_future_usd':dollars(Decimal(inp['outstanding_usd'])+Decimal(inp['reserved_remaining_usd'])+committed_elapsed),
          'hourly_usd':inp['hourly_upper_usd'],'fixed_remaining_usd':'0','phases':phases}
    numerical=forecast(base)
    total_ms=sum(p['remaining_ms_with_margin'] for p in numerical['phases'].values())
    numerical_cost=Decimal(total_ms)*rate/3600000
    remaining=Decimal(plan['maximum_charge_micro_usd'])/10**6-committed_elapsed
    if numerical_cost>remaining:raise EvidenceError('complete training and replay exceed remaining original rental allowance')
    # Reserve the entire original lifetime remainder, including unused work time,
    # setup, publications, export, termination grace and billing slack. This is a
    # conservative envelope, explicitly NOT an estimate of measured fixed cost.
    r['forecast_input']={**base,'fixed_remaining_usd':dollars(remaining-numerical_cost)}
    projected=forecast(r['forecast_input'])
    if projected['result']!='FITS_OPERATING_LIMIT':raise EvidenceError('original aggregate operating limit exceeded')
    return r,{'schema':'ovl.registration-forecast-envelope.v1','forecast':projected,
              'original_maximum_rental_micro_usd':plan['maximum_charge_micro_usd'],
              'elapsed_upper_usd':dollars(elapsed),'remaining_lifetime_upper_usd':dollars(remaining),
              'construction_exposure_seconds':construction_seconds,'elapsed_and_construction_upper_usd':dollars(committed_elapsed),
              'scope':'measured pilot rates with required margins, bounded by original whole-rental ceiling; fixed costs not claimed measured'}


def production_job(kind,profile,offline_config,static_files,packet,bundle,production_policy,source_policy,
                   *,record_files=None):
    if kind not in ('production-record','full-replay'):raise EvidenceError('unsupported production stage')
    base=profile['remote_root'];name='production-record' if kind=='production-record' else 'production-replay'
    remote=base+'-'+name;record=base+'-production-record';audit=base+'/control-'+name
    public=base+'/production-inputs';source=base+'/inputs/source';prepared=base+'/prepared'
    r=read_json(packet/'registration.json')
    values={'--packet':public+'/packet','--registration-bundle':public+'/registration.sigstore.json',
            '--production-policy':public+'/production-policy.json','--source-policy':public+'/source-policy.json',
            '--source-checkout':source,'--wikipedia-stream':prepared+'/wikipedia','--conversation-stream':prepared+'/conversation',
            '--output':remote,'--progress-policies':record+'/external-progress-policies.json'}
    if kind=='production-record':values.update({'--registration-sha256':digest(r),'--checkpoint-deadline':DEADLINE,
        '--key-directory':base+'/private/run-key','--anchor-directory':record+'/public-anchors'})
    else:values.update({'--chain-directory':record,'--progress-directory':record+'/public-anchors'})
    inputs=list(deepcopy(static_files))
    def add(path,local):inputs.append({'path':path,'bytes':local.stat().st_size,'sha256':file_hash(local)})
    from ovl_pipeline.production_anchoring import PACKET_FILES
    for entry in sorted(PACKET_FILES):add(public+'/packet/'+entry,packet/entry)
    for entry,local in [('registration.sigstore.json',bundle),('production-policy.json',production_policy),('source-policy.json',source_policy)]:
        add(public+'/'+entry,local)
    if kind=='full-replay':
        if record_files is None:raise EvidenceError('full replay requires complete retained recorded input inventory')
        inputs.extend({**f,'path':record+'/'+f['path']} for f in record_files)
    elif record_files is not None:raise EvidenceError('fresh recording cannot load recorded state')
    if len({f['path'] for f in inputs})!=len(inputs):raise EvidenceError('duplicate independently selected production inputs')
    return {'schema':'ovl.pod-job.v1','kind':kind,'cwd':base,'deadline_epoch':DEADLINE,'stop_grace_seconds':15,
      'minimum_free_bytes':16*1024**3,'required_files':inputs,'export_roots':[remote,audit],
      'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','OVL_ACTIVITY_FILE':audit+'/activity.json'},
      'argv':[base+'/runtime/public-python/python/bin/python3.12','-I','-S',base+'/inputs/pod_runtime_setup.py','launch',
              '--config',base+'/inputs/offline-config.json','--config-sha256',file_hash(offline_config),
              '--inputs',base+'/inputs','--runtime',base+'/runtime','--output',audit+'/audit',
              '--module','ovl_pipeline.'+('production_record' if kind=='production-record' else 'production_replay'),
              '--',*[v for item in values.items() for v in item]]}
