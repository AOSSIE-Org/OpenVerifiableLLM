"""Closed production recipe/evidence graph checks; not execution admission.

This module performs no signing, GPU work or network verification. Callers still
must authenticate each public parent and actually reconstruct/replay as required.
"""
import re
from math import gcd

from . import schema
from .budget import forecast,ceil_div
from .canonical import EvidenceError,digest,require_digest

VERIFIER_POLICY={
    'schema':'ovl.production-verifier-profile.v1',
    'raw_reconstruction':'all-transformations-fresh-output',
    'initialization':'regenerated-never-load-prover-state',
    'replay':'continuous-all-updates-byte-exact',
    'state':'parameters-buffers-optimizer-rng-control',
    'anchoring':'registration-and-every-primary-boundary-before-next-update',
    'publication':'full-anonymous-download-at-immutable-revision',
    'performed_by':'project-operator',
    'independent_third_party':False,
}


def identifier(value,label):
    if type(value) is not str or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',value):
        raise EvidenceError('invalid '+label)


def checkpoint_count(updates,primary,recovery):
    # Union of completed-update intervals plus final tail. Initial/transition
    # checkpoints are setup work reserved in fixed_remaining_usd, like pilot setup.
    common=primary*recovery//gcd(primary,recovery)
    regular=updates//primary+updates//recovery-updates//common
    return regular+int(updates%primary!=0 and updates%recovery!=0)


def validate(value):
    schema.fields(value,'schema scope run_id attempt_id code_revision code_root source_statement_sha256 source_bundle_sha256 source_policy_sha256 preparation_sha256 recipe kernel runtime initialization run_public_key coverage recovery_every pilots forecast_input verifier_policy conversation_policy','production registration')
    if value['schema']!='ovl.production-registration.v1' or value['scope']!='complete-wikipedia-and-public-conversation':
        raise EvidenceError('unsupported production registration')
    for name in ('run_id','attempt_id'):identifier(value[name],name)
    if type(value['code_revision']) is not str or not re.fullmatch('[0-9a-f]{40}',value['code_revision']):
        raise EvidenceError('invalid production source revision')
    for name in ('code_root','source_statement_sha256','source_bundle_sha256','source_policy_sha256','preparation_sha256','run_public_key'):
        require_digest(value[name])
    recipe=value['recipe'];schema.recipe(recipe,gpu=True)
    schema.fields(value['kernel'],'schema precision','production kernel')
    if value['kernel']['schema']!='ovl.gpu-kernel.v1' or value['kernel']['precision'] not in ('fp32','bf16'):
        raise EvidenceError('unsupported production precision')
    runtime=value['runtime']
    schema.fields(runtime,'container_image dependency_lock_sha256 compatible_environment_sha256','production runtime')
    if type(runtime['container_image']) is not str or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9./_-]*@sha256:[0-9a-f]{64}',runtime['container_image']):
        raise EvidenceError('immutable container image required')
    for name in ('dependency_lock_sha256','compatible_environment_sha256'):require_digest(runtime[name])
    init=value['initialization'];schema.fields(init,'state_sha256 regeneration_report_sha256 warmup_updates','production initialization')
    for name in ('state_sha256','regeneration_report_sha256'):require_digest(init[name])
    schema.integer(init['warmup_updates'],1,1000,'discarded warmup count')
    schema.integer(value['recovery_every'],1,recipe['boundary_every'],'recovery checkpoint interval')
    if value['verifier_policy']!=VERIFIER_POLICY or value['conversation_policy']!='one-epoch-reset-adamw-v1':
        raise EvidenceError('full reconstruction/replay/phase policy required')
    if set(value['coverage'])!={'wikipedia','conversation'} or set(value['pilots'])!=set(value['coverage']):
        raise EvidenceError('both full phases and pilots required')
    inp=value['forecast_input']
    if inp.get('schema')!='ovl.cost-forecast-input.v3':raise EvidenceError('production requires v3 cost inputs')
    projected=forecast(inp)
    if projected['result']!='FITS_OPERATING_LIMIT':raise EvidenceError('full work exceeds operating budget')
    primary_count=2
    for phase,census in value['coverage'].items():
        schema.fields(census,'schema scope phase stream_sha256 recipe_sha256 documents targets target_bearing_windows updates full_batch_updates final_batch_rows context batch_size padded_positions masked_context_positions training_coverage','complete production census')
        if (census['schema']!='ovl.complete-schedule-counts.v1' or census['scope']!='complete-stream-census'
                or census['phase']!=phase or census['training_coverage']!='NOT_RUN'):
            raise EvidenceError('invalid preproduction complete census')
        require_digest(census['stream_sha256']);require_digest(census['recipe_sha256'])
        for name in ('documents','targets','target_bearing_windows','updates','final_batch_rows','context','batch_size'):
            schema.integer(census[name],1,2**53-1,name)
        for name in ('full_batch_updates','padded_positions','masked_context_positions'):
            schema.integer(census[name],0,2**53-1,name)
        windows=census['target_bearing_windows'];batch=recipe['batch_size'];context=recipe['context']
        if (census['recipe_sha256']!=digest(recipe) or census['batch_size']!=batch or census['context']!=context
                or census['updates']!=ceil_div(windows,batch) or census['full_batch_updates']!=windows//batch
                or census['final_batch_rows']!=(windows-1)%batch+1
                or windows*context!=census['targets']+census['padded_positions']+census['masked_context_positions']):
            raise EvidenceError('inconsistent complete census arithmetic')
        primary_count+=ceil_div(census['updates'],recipe['boundary_every'])
        pilot=value['pilots'][phase];schema.fields(pilot,'record_sha256 replay_sha256','pilot parents')
        for root in pilot.values():require_digest(root)
        f=inp['phases'][phase]
        if (f['schedule_sha256']!=digest(census) or f['recipe_sha256']!=digest(recipe)
                or f['stream_sha256']!=census['stream_sha256'] or f['updates']!=census['updates']
                or f['measurement_sha256']!=pilot['record_sha256'] or f['replay_sha256']!=pilot['replay_sha256']
                or f['training_completed']!=0 or f['replay_completed']!=0
                or f['production_checkpoint_every']!=value['recovery_every']
                or f['production_checkpoints']<checkpoint_count(census['updates'],recipe['boundary_every'],value['recovery_every'])):
            raise EvidenceError('forecast not bound to complete registration work')
    if primary_count>4096:raise EvidenceError('production chain exceeds bounded format')
    return {'schema':'ovl.production-contract-check.v1','result':'PASS','registration_sha256':digest(value),
            'scope':'structure-parent-digests-and-budget-arithmetic-only','primary_boundaries':primary_count,
            'forecast':projected,'publisher_identity':'NOT_RUN','raw_reconstruction':'NOT_RUN',
            'gpu_reproducibility':'NOT_RUN','initial_state_regeneration':'NOT_RUN','provider_guard':'NOT_RUN',
            'fixed_cost_basis':'NOT_RUN','container_identity':'NOT_RUN','installed_runtime_identity':'NOT_RUN',
            'production_admission':'NOT_RUN'}
