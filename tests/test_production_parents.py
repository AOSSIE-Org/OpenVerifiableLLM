"""Synthetic report consistency tests; no authenticated publisher/GPU/training claims."""
import copy
import pytest
from test_production_contract import registration
from ovl_pipeline.production_parents import validate_parents
from ovl_pipeline.canonical import EvidenceError,digest


def parents():
    r=registration();kernel=r['kernel'];env={'compatible':{'schema':'ovl.gpu-environment.v1','kernel':kernel},'reproducibility_validation':'NOT_RUN','production_admission':'NOT_RUN'}
    r['runtime']['compatible_environment_sha256']=digest(env['compatible'])
    source={'schema':'ovl.source-preparation.v2','scope':'production-source-preparation','run_id':r['run_id'],
            'source_revision':'1'*40,'code':['synthetic-code-pin'],'environment':{'synthetic':True},
            'conversation':{'inventory':['synthetic-inventory'],'splits':{'train':'train.parquet','validation':'validation.parquet'}}}
    r['source_statement_sha256']=digest(source)
    policy={'statement_sha256':digest(source),'source_revision':source['source_revision']};r['source_policy_sha256']=digest(policy)
    streams={p:{'documents':10,'targets':48000,'phase':'wikipedia' if p=='wikipedia' else 'conversation','index_root':digest(p)} for p in ('wikipedia','conversation','conversation-validation')}
    prepared={'schema':'ovl.complete-preparation.v1','source_commitment_sha256':digest(source),
        'code':source['code'],'environment':source['environment'],'validation_used_for_training':False,
        'tokenizer':{'vocab_size':320},'streams':streams,
        'conversation_selection':{'source_inventory_root':digest(source['conversation']['inventory']),'split_filenames':source['conversation']['splits']}}
    r['preparation_sha256']=digest(prepared)
    initial={'scope':'preproduction-regenerated-initial-state','result':'RECORDED_AWAITING_FRESH_REGENERATION','warmup_weights_discarded':True,'schema':'ovl.initialization-record.v1','recipe':r['recipe'],'kernel':kernel,'code_root':r['code_root'],
        'checkpoint':{'schema':'ovl.checkpoint.v1','state_root':'9'*64,'files':[{'path':n,'bytes':1,'sha256':'8'*64} for n in ('state.json','state.safetensors')]},'warmup_updates':4,'stream_sha256':digest(streams['wikipedia']),
        'process_observation':{'pid':1,'boot_id':'00000000-0000-0000-0000-000000000001','start_ticks':'10'},'environment':env,
        'control':{'phase':'wikipedia','global_step':0,'phase_step':0,'cursor':0,'transcript':'a'*64,'schedule':'constant-lr-v1','accumulation':'none','scaler':'none'},
        'parameter_count':100,'production_admission':'NOT_RUN'}
    verified={'schema':'ovl.initialization-verification.v1','result':'PASS','scope':'complete-initial-state-regenerated-and-compared',
        'prover_tensors_loaded_as_state':False,'distinct_process_from_record':True,'record_sha256':digest(initial),
        'code_root':r['code_root'],'recipe_sha256':digest(r['recipe']),'initial_state_sha256':'9'*64,
        'warmup_updates':4,'stream_sha256':digest(streams['wikipedia']),
        'process_observation':{'pid':2,'boot_id':'00000000-0000-0000-0000-000000000001','start_ticks':'20'},'environment':env,
        'process_identity_scope':'operator OS observation, not hardware attestation',
        'performed_by':'project-operator','independent_third_party':False,'production_admission':'NOT_RUN'}
    r['initialization']['regeneration_report_sha256']=digest(verified)
    records={};replays={}
    for phase in ('wikipedia','conversation'):
        census=r['coverage'][phase];census['stream_sha256']=digest(streams[phase])
        settings={'schema':'ovl.gpu-pilot-settings.v1','scope':'development-gpu-pilot-only','recipe':r['recipe'],
            'kernel':kernel,'stream':streams[phase],'code_root':r['code_root'],'warmup_updates':4,'environment':env,
            'checkpoint_every':10,'requested_updates':None,'requested_seconds':600}
        boundaries=[];previous=digest(settings)
        for index,step in enumerate(range(0,101,10)):
            b={'index':index,'step':step,'control':{'global_step':step,'phase':phase,'phase_step':step,'cursor':step*48,'transcript':digest(step),'schedule':'constant-lr-v1','accumulation':'none','scaler':'none','pilot_cycle':0},'path':f'boundary-{index:05d}',
               'checkpoint':{'state_root':digest({'synthetic-step':step})},'previous':previous}
            boundaries.append(b);previous=digest(b)
        record={'schema':'ovl.gpu-pilot-record.v1','scope':'development-gpu-pilot-only','settings':settings,'updates':100,
            'boundaries':boundaries,'measured_full_batch_updates':100,'measured_ms':600000,'eligible_duration_for_forecast':True,
            'timed_checkpoints':10,'warmup_excluded':True,'overhead_included':True,'measured_targets':4800,
            'setup_including_warmup_ms':100000}
        replay={'schema':'ovl.gpu-pilot-replay.v1','scope':'fresh-initialization-continuous-pilot-replay','result':'PASS',
            'verifier_checkpoint_overhead_included':True,'verifier_checkpoints_saved':len(boundaries),
            'resume_from':None,'initial_state_regenerated':True,'record_sha256':digest(record),'environment':env,
            'updates_recomputed':100,'compared':[{'index':b['index'],'state_root':b['checkpoint']['state_root']} for b in boundaries],
            'measured_ms':610000,'eligible_for_forecast_comparison':True,'measured_targets':4800,'measured_full_batch_updates':100,'timed_checkpoints':10,
            'setup_including_warmup_ms':100000}
        records[phase]=record;replays[phase]=replay
        r['pilots'][phase]={'record_sha256':digest(record),'replay_sha256':digest(replay)}
        r['forecast_input']['phases'][phase].update(schedule_sha256=digest(census),stream_sha256=digest(streams[phase]),measurement_sha256=digest(record),replay_sha256=digest(replay))
    return r,dict(source=source,source_policy=policy,prepared=prepared,initial_record=initial,initial_verification=verified,pilot_records=records,pilot_replays=replays)


def rebind(r,objects):
    """Give adversarial consistency checks fully rehashed parents, not trivial stale hashes."""
    init=objects['initial_record'];iv=objects['initial_verification'];iv['record_sha256']=digest(init)
    r['initialization']['regeneration_report_sha256']=digest(iv)
    r['source_statement_sha256']=digest(objects['source']);r['preparation_sha256']=digest(objects['prepared'])
    for phase in ('wikipedia','conversation'):
        record=objects['pilot_records'][phase];replay=objects['pilot_replays'][phase];replay['record_sha256']=digest(record)
        r['pilots'][phase]={'record_sha256':digest(record),'replay_sha256':digest(replay)}
        r['forecast_input']['phases'][phase].update(measurement_sha256=digest(record),replay_sha256=digest(replay))


def test_synthetic_parent_consistency_cannot_authenticate_assertions_or_admit_execution():
    r,p=parents();v=validate_parents(r,**p)
    assert v['result']=='PASS' and v['assertion_truth_established'] is False and v['production_admission']=='NOT_RUN'

@pytest.mark.parametrize('change',['source-run','validation','conversation-parent','vocab','init-loaded','init-same-process',
    'init-state','init-runtime','pilot-recipe','pilot-stream','sampled-replay','resume-replay','empty-boundaries',
    'boundary-parent','false-rate','false-replay-time','fixed-pilot'])
@pytest.mark.parametrize('version',['v3','v4'])
def test_rehashed_parents_cannot_hide_inconsistent_evidence(change,version):
    r,p=parents();record=p['pilot_records']['wikipedia'];replay=p['pilot_replays']['wikipedia']
    r['forecast_input']['schema']='ovl.cost-forecast-input.'+version
    if change=='source-run':p['source']['run_id']='other-run'
    elif change=='validation':p['prepared']['validation_used_for_training']=True
    elif change=='conversation-parent':p['prepared']['conversation_selection']['source_inventory_root']='0'*64
    elif change=='vocab':p['prepared']['tokenizer']['vocab_size']=321
    elif change=='init-loaded':p['initial_verification']['prover_tensors_loaded_as_state']=True
    elif change=='init-same-process':p['initial_verification']['process_observation']=p['initial_record']['process_observation']
    elif change=='init-state':p['initial_record']['checkpoint']['state_root']='0'*64
    elif change=='init-runtime':p['initial_record']['environment']={'compatible':{'test_runtime':'CPU-substitute'}}
    elif change=='pilot-recipe':record['settings']['recipe']={**r['recipe'],'seed':99}
    elif change=='pilot-stream':record['settings']['stream']={**record['settings']['stream'],'targets':1}
    elif change=='sampled-replay':replay['updates_recomputed']=1
    elif change=='resume-replay':replay['resume_from']=1
    elif change=='empty-boundaries':record['boundaries']=[];replay['compared']=[]
    elif change=='boundary-parent':record['boundaries'][1]['previous']='0'*64
    elif change=='false-rate':record['measured_ms']=300000
    elif change=='false-replay-time':replay['measured_ms']=1
    else:record['settings']['requested_updates']=100;record['settings']['requested_seconds']=None
    rebind(r,p)
    with pytest.raises(EvidenceError):validate_parents(r,**p)

@pytest.mark.parametrize('change',['validation-alias','retained-warmup','wrong-init-scope','wrong-init-result','missing-pilot-cycle','false-pilot-targets'])
def test_review_report_counterexamples(change):
    r,p=parents();record=p['pilot_records']['wikipedia']
    if change=='validation-alias':p['prepared']['streams']['conversation-validation']=p['prepared']['streams']['conversation'].copy()
    elif change=='retained-warmup':p['initial_record']['warmup_weights_discarded']=False
    elif change=='wrong-init-scope':p['initial_record']['scope']='imported-pretrained-state'
    elif change=='wrong-init-result':p['initial_record']['result']='FAILED'
    elif change=='missing-pilot-cycle':record['boundaries'][0]['control'].pop('pilot_cycle')
    else:record['measured_targets']+=1;p['pilot_replays']['wikipedia']['measured_targets']+=1
    rebind(r,p)
    with pytest.raises(EvidenceError):validate_parents(r,**p)


@pytest.mark.parametrize('change',['missing','false','count'])
def test_replay_forecast_requires_actual_verifier_checkpoint_overhead(change):
    r,p=parents();replay=p['pilot_replays']['wikipedia']
    if change=='missing':del replay['verifier_checkpoint_overhead_included']
    elif change=='false':replay['verifier_checkpoint_overhead_included']=False
    else:replay['verifier_checkpoints_saved']-=1
    rebind(r,p)
    with pytest.raises(EvidenceError,match='verifier checkpoint'):validate_parents(r,**p)


@pytest.mark.parametrize('damage',[None,'missing-replay-delivery','wrong-phase','unknown-settings','legacy-hidden-delivery'])
def test_delivered_pilot_reports_preserve_original_forecast_and_parent_gates(damage):
    r,p=parents();record=p['pilot_records']['wikipedia'];replay=p['pilot_replays']['wikipedia']
    policy={'schema':'ovl.pilot-delivery-policy.v1','session':'a'*64,'mode':'record','phase':'wikipedia',
            'deadline_epoch':2000,'copy_timeout_seconds':120,'maximum_checkpoint_bytes':10*1024**2}
    record['settings'].update(schema='ovl.gpu-pilot-settings.v2',delivery=policy)
    replay.update(schema='ovl.gpu-pilot-replay.v2',delivery={**policy,'mode':'replay','session':'b'*64})
    if damage=='missing-replay-delivery':replay['schema']='ovl.gpu-pilot-replay.v1';replay.pop('delivery')
    elif damage=='wrong-phase':replay['delivery']['phase']='conversation'
    elif damage=='unknown-settings':record['settings']['schema']='ovl.gpu-pilot-settings.v3'
    elif damage=='legacy-hidden-delivery':record['settings']['schema']='ovl.gpu-pilot-settings.v1'
    previous=digest(record['settings'])
    for b in record['boundaries']:b['previous']=previous;previous=digest(b)
    rebind(r,p)
    if damage is None:assert validate_parents(r,**p)['production_admission']=='NOT_RUN'
    else:
        with pytest.raises(EvidenceError):validate_parents(r,**p)
