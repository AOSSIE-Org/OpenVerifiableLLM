"""Production structure/budget binding only; synthetic reports never admit a run."""
import copy
import pytest
from ovl_pipeline.fixture import recipe
from ovl_pipeline.canonical import EvidenceError,digest
from ovl_pipeline.production_contract import validate,checkpoint_count,VERIFIER_POLICY


def registration():
    r=recipe(320);r['boundary_every']=30
    v=dict(schema='ovl.production-registration.v1',scope='complete-wikipedia-and-public-conversation',
        run_id='synthetic-structure-check',attempt_id='not-a-real-run',code_revision='1'*40,code_root='2'*64,
        source_statement_sha256='3'*64,source_bundle_sha256='f'*64,source_policy_sha256='4'*64,preparation_sha256='5'*64,
        recipe=r,kernel={'schema':'ovl.gpu-kernel.v1','precision':'bf16'},
        runtime={'container_image':'runpod/pytorch@sha256:'+'6'*64,'dependency_lock_sha256':'7'*64,'compatible_environment_sha256':'8'*64},
        initialization={'state_sha256':'9'*64,'regeneration_report_sha256':'a'*64,'warmup_updates':4},
        run_public_key='b'*64,coverage={},recovery_every=10,pilots={},
        forecast_input={'schema':'ovl.cost-forecast-input.v3','spent_usd':'1','committed_future_usd':'1','fixed_remaining_usd':'5','hourly_usd':'0.99','phases':{}},
        verifier_policy=copy.deepcopy(VERIFIER_POLICY),conversation_policy='one-epoch-reset-adamw-v1')
    for phase in ('wikipedia','conversation'):
        c=dict(schema='ovl.complete-schedule-counts.v1',scope='complete-stream-census',phase=phase,
            stream_sha256='c'*64,recipe_sha256=digest(r),documents=10,targets=48000,
            target_bearing_windows=3000,updates=1000,full_batch_updates=1000,final_batch_rows=3,
            context=16,batch_size=3,padded_positions=0,masked_context_positions=0,training_coverage='NOT_RUN')
        v['coverage'][phase]=c;v['pilots'][phase]=dict(record_sha256='d'*64,replay_sha256='e'*64)
        v['forecast_input']['phases'][phase]=dict(updates=1000,training_completed=0,replay_completed=0,
            measured_full_batch_updates=100,measured_ms=600000,warmup_excluded=True,overhead_included=True,
            measurement_sha256='d'*64,recipe_sha256=digest(r),stream_sha256='c'*64,schedule_sha256=digest(c),
            replay_sha256='e'*64,eligible_duration_for_forecast=True,measured_updates=100,
            replay_measured_ms=610000,measured_checkpoints=10,measured_checkpoint_every=10,
            production_checkpoint_every=10,production_checkpoints=100)
    return v


def test_contract_pass_does_not_admit_gpu_execution_or_attest_evidence():
    result=validate(registration())
    assert result['result']=='PASS' and result['primary_boundaries']==70
    for k in ('publisher_identity','raw_reconstruction','gpu_reproducibility','initial_state_regeneration','provider_guard','production_admission'):
        assert result[k]=='NOT_RUN'

@pytest.mark.parametrize('change',['sample','unpriced-recovery','wrong-pilot','wrong-stream','recipe','missing-phase','tag-image','already-trained','budget','unknown','skip-replay'])
def test_production_contract_rejects_omitted_or_inconsistent_work(change):
    v=registration();f=v['forecast_input']['phases']['wikipedia']
    if change=='sample':v['coverage']['wikipedia']['scope']='sample'
    elif change=='unpriced-recovery':v['recovery_every']=5
    elif change=='wrong-pilot':v['pilots']['wikipedia']['record_sha256']='f'*64
    elif change=='wrong-stream':v['coverage']['wikipedia']['stream_sha256']='f'*64
    elif change=='recipe':v['recipe']['batch_size']=4
    elif change=='missing-phase':del v['coverage']['conversation']
    elif change=='tag-image':v['runtime']['container_image']='runpod/pytorch:latest'
    elif change=='already-trained':f['training_completed']=1
    elif change=='budget':v['forecast_input']['spent_usd']='89'
    elif change=='unknown':v['allow_unverified']=True
    else:v['verifier_policy']['replay']='sampled'
    with pytest.raises(EvidenceError):validate(v)

@pytest.mark.parametrize('updates,primary,recovery',[(1,30,10),(30,30,10),(31,30,10),(999,30,7),(1000,30,10)])
def test_recovery_and_primary_union_counts_every_completed_checkpoint(updates,primary,recovery):
    # Independently enumerate a tiny schedule to challenge arithmetic, including
    # nondividing intervals and final tails. Initial/setup state is separate.
    expected={n for n in range(1,updates+1) if n%primary==0 or n%recovery==0}|{updates}
    assert checkpoint_count(updates,primary,recovery)==len(expected)
