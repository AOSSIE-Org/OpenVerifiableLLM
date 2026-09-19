#!/usr/bin/env python3
"""Join separate, retained operator executions without claiming a new replay.

Freshly authenticate all public boundaries, check complete retained execution and
safe-state bytes, and run all held-out evaluation/inference checks. The original
clean reconstruction and continuous replay remain explicitly separate executions.
The strict fresh-work production_verify command remains unchanged.
"""
from pathlib import Path
import time

from ovl_pipeline.canonical import (EvidenceError, canonical, confined, digest, file_hash,
    inventory, parse_json, read_json, sha256, verify_inventory, write_json)
from ovl_pipeline.schema import fields, integer
from ovl_pipeline.production_composition import public_reference, validate as validate_composition
from ovl_pipeline.production_verify import STAGES, evaluate
from ovl_pipeline.state import read_state, unpack
from production_retention import verify as verify_retention


def complete_tree(root, files):
    root=Path(root)
    if any(p.is_symlink() for p in [root,*root.absolute().parents]) or not root.is_dir():
        raise EvidenceError('regular complete artifact tree required')
    actual=[]
    for path in root.rglob('*'):
        if path.is_symlink() or not(path.is_file() or path.is_dir()):raise EvidenceError('unsafe retained artifact')
        if path.is_file():actual.append(path.relative_to(root).as_posix())
    if sorted(actual)!=[f['path'] for f in files]:raise EvidenceError('complete artifact inventory differs')
    verify_inventory(root,files)


def public_object(reference):
    from ovl_pipeline.source_commitment import fetch_metadata
    public_reference(reference);data=fetch_metadata(reference['url'])
    if sha256(data)!=reference['sha256']:raise EvidenceError('actual public execution evidence differs')
    return parse_json(data,canonical_required=True)


def prepared_bytes(directory, r):
    """Recheck every byte against the registered manifest; reuse completed logic."""
    from ovl_pipeline.prepared_verification import FILES, STAGES as MAPPING
    value=read_json(directory/'preparation.json')
    if digest(value)!=r['preparation_sha256'] or value['source_commitment_sha256']!=r['source_statement_sha256']:
        raise EvidenceError('reconstructed preparation belongs to another registration')
    expected=[{'path':'preparation.json','bytes':len(canonical(value)),'sha256':digest(value)}]
    for stage, name in FILES.items():
        manifest=value[MAPPING[stage]] if stage in MAPPING else value['streams'][stage]
        expected.append({'path':stage+'/'+name,'bytes':len(canonical(manifest)),'sha256':digest(manifest)})
        expected.extend({**f,'path':stage+'/'+f['path']} for f in manifest['files'])
    expected.sort(key=lambda f:f['path']);complete_tree(directory,expected)
    return {'schema':'ovl.reconstructed-bytes-rechecked.v1','result':'PASS','preparation_sha256':digest(value),
            'files':expected,'scope':'all registered reconstructed bytes rehashed; earlier full transformation execution reused'}


def reconstruction(checkpoint, observation, r):
    if (checkpoint.get('schema')!='ovl.complete-clean-reconstruction-checkpoint.v2' or checkpoint.get('result')!='PASS'
        or checkpoint.get('independent_third_party') is not False):raise EvidenceError('selected clean reconstruction evidence required')
    result=checkpoint['report']
    if (result['result']!='PASS' or result['scope']!='complete-source-preparation'
        or result['preparation_sha256']!=r['preparation_sha256'] or result['source_commitment_sha256']!=r['source_statement_sha256']
        or result['full_reconstruction_compared'] is not True or result['stages_executed_this_run']!=STAGES
        or result['stages_adopted_from_local_cache']!=[] or result['execution_observation_sha256']!=digest(observation)):
        raise EvidenceError('earlier execution did not reconstruct all registered transformations')
    if (observation.get('schema')!='ovl.preparation-execution.v1' or observation.get('status')!='PASS'
        or observation.get('source_commitment_sha256')!=r['source_statement_sha256']
        or observation.get('resume_requested') is not False or observation.get('stages_executed_this_run')!=STAGES
        or observation.get('stages_adopted_from_local_cache')!=[]):
        raise EvidenceError('earlier process observation contradicts clean reconstruction')
    integer(checkpoint['whole_job_elapsed_ms'],1,2**53-1,'original complete reconstruction duration')
    return result


def replay_states(r, envelopes, record, directory, value):
    """Read every primary and recovery safe state; do not restore any of them."""
    session=read_json(directory/'session.json')
    if (session.get('schema')!='ovl.production-replay-session.v1'
        or session.get('scope')!='fresh-start-continuous-all-updates-numerical-replay'
        or digest(session)!=value['session_sha256'] or session.get('registration_sha256')!=digest(r)
        or session.get('chain_sha256')!=digest(envelopes) or session.get('code_root')!=r['code_root']
        or session.get('prover_state_restored') is not False or session.get('resume_supported') is not False
        or digest(session['environment']['compatible'])!=r['runtime']['compatible_environment_sha256']):
        raise EvidenceError('retained replay session differs from selected full execution')
    if (value.get('result')!='PASS' or value.get('registration_sha256')!=digest(r)
        or value.get('chain_sha256')!=digest(envelopes) or value.get('initial_state_regenerated') is not True
        or value.get('prover_checkpoints_restored') is not False
        or value.get('updates_recomputed')!={p:r['coverage'][p]['updates'] for p in r['coverage']}
        or value.get('targets_recomputed')!={p:r['coverage'][p]['targets'] for p in r['coverage']}
        or len(value['comparisons'])!=len(envelopes)):
        raise EvidenceError('retained replay does not cover all updates and targets')
    def state(path, expected, control=None):
        marker=read_json(path/'checkpoint.json')
        if marker['state_root']!=expected:raise EvidenceError('retained safe state root differs')
        metadata,tensors=read_state(path,marker)
        if control is not None and unpack(metadata['tree'],tensors)['control']!=control:
            raise EvidenceError('retained safe state control differs')
        return marker
    for envelope,comparison in zip(envelopes,value['comparisons']):
        body=envelope['body'];index=body['index'];expected=body['checkpoint']['state_root']
        if (comparison['index']!=index or comparison['boundary_sha256']!=digest(envelope)
            or comparison['state_root']!=expected or comparison['control']!=body['control'] or comparison['result']!='PASS'):
            raise EvidenceError('retained comparison differs from public primary boundary')
        if state(record/body['checkpoint_path'],expected,body['control'])!=body['checkpoint']:
            raise EvidenceError('recorded primary marker differs from public commitment')
        if state(directory/f'verifier-boundary-{index:05d}',expected,body['control'])!=comparison['verifier_checkpoint']:
            raise EvidenceError('replay primary marker differs from comparison')
    expected_steps=[];offset=0
    for phase in ('wikipedia','conversation'):
        total=r['coverage'][phase]['updates']
        expected_steps.extend(offset+n for n in range(r['recovery_every'],total,r['recovery_every'])
                              if n%r['recipe']['boundary_every'])
        offset+=total
    if [v['global_step'] for v in value['recovery_checkpoints']]!=expected_steps:
        raise EvidenceError('retained replay omits or reorders recovery checkpoints')
    for recovery in value['recovery_checkpoints']:
        step=recovery['global_step'];expected=recovery['state_root']
        state(record/f'recovery-{step:09d}',expected)
        state(directory/f'verifier-recovery-{step:09d}',expected)
    return len(envelopes)+len(expected_steps)


def process(r, audit, job, terminal, remote_output):
    receipt=read_json(audit/'process.json');launch=read_json(audit/'launch.json')
    if (receipt.get('schema')!='ovl.audited-runtime-process.v1' or receipt.get('exit_code')!=0
        or type(receipt.get('exit_code')) is not int or receipt['launch_sha256']!=digest(launch)
        or terminal.get('state')!='EXITED' or terminal.get('exit_code')!=0
        or terminal.get('job_sha256')!=digest(job) or job['kind']!='full-replay'
        or launch.get('schema')!='ovl.audited-runtime-launch.v1'
        or launch.get('module')!='ovl_pipeline.production_replay'
        or launch['dependency_lock_sha256']!=r['runtime']['dependency_lock_sha256']):
        raise EvidenceError('actual successful audited replay process required')
    args=launch['arguments']
    if type(args) is not list or len(args)%2 or len(set(args[::2]))!=len(args)//2:
        raise EvidenceError('ambiguous audited replay arguments')
    selected=dict(zip(args[::2],args[1::2]))
    required_flags={'--packet','--registration-bundle','--production-policy','--source-policy','--source-checkout',
                    '--chain-directory','--progress-directory','--progress-policies','--wikipedia-stream',
                    '--conversation-stream','--output'}
    if set(selected)!=required_flags or launch.get('source')!=selected['--source-checkout']+'/src':
        raise EvidenceError('complete independently selected replay arguments/source required')
    if selected.get('--output')!=remote_output or remote_output not in job['export_roots']:
        raise EvidenceError('audited replay output differs from retained worker root')
    inputs={f['path']:f for f in job['required_files']}
    if len(inputs)!=len(job['required_files']):raise EvidenceError('duplicate replay input selection')
    expected={selected['--packet']+'/registration.json':digest(r),
              selected['--packet']+'/source-statement.json':r['source_statement_sha256'],
              selected['--packet']+'/source-statement.sigstore.json':r['source_bundle_sha256'],
              selected['--packet']+'/preparation.json':r['preparation_sha256']}
    expected.update({selected['--'+p+'-stream']+'/stream.json':r['coverage'][p]['stream_sha256']
                     for p in ('wikipedia','conversation')})
    if any(name not in inputs or inputs[name]['sha256']!=root for name,root in expected.items()):
        raise EvidenceError('audited replay input selection differs from registered parents')
    argv=job['argv']
    if (argv.count('--module')!=1 or argv[argv.index('--module')+1]!=launch['module']
        or '--' not in argv or argv[argv.index('--')+1:]!=args):
        raise EvidenceError('worker did not select this exact audited replay invocation')
    manifest=read_json(audit/'wheel-payloads.json');installed=read_json(audit/'installed-audit.json')
    payloads=read_json(audit/'python-payloads.json');python=read_json(audit/'python-audit.json');origin=launch['interpreter_origin']
    if (digest(manifest)!=launch['wheel_manifest_sha256'] or digest(installed)!=launch['installed_audit_sha256']
        or installed.get('result')!='PASS' or installed['wheel_manifest_sha256']!=digest(manifest)
        or manifest['dependency_lock_sha256']!=r['runtime']['dependency_lock_sha256']
        or origin is None or digest(payloads)!=origin['manifest_sha256'] or digest(python)!=origin['audit_sha256']
        or python.get('result')!='PASS' or python['archive_manifest_sha256']!=digest(payloads)
        or python['archive_sha256']!=origin['archive_sha256'] or payloads['archive_sha256']!=origin['archive_sha256']):
        raise EvidenceError('retained complete runtime audits do not match replay process')
    return receipt,launch


def assemble(*,packet,bundle,production_policy,source_policy,source_checkout,chain,progress,progress_policies,
             raw,prepared,exports,context,prior_checkpoint,prior_observation,prior_public,replay_public,
             control,transports,job_file,job_sha256,worker_sha256,retention,retention_sha256,
             replay_root,audit_root,audit_relative,output):
    from ovl_pipeline.production_replay import authenticate
    from ovl_pipeline.production_export import verify_replayed_exports
    from ovl_pipeline.production_release import reports
    from ovl_pipeline.training import code_root
    if output.exists():raise EvidenceError('assembly requires fresh output; preserve previous evidence')
    started=time.monotonic_ns()
    r,envelopes,endorsements=authenticate(packet,bundle,production_policy,source_policy,source_checkout,chain,progress,progress_policies)
    if code_root()!=r['code_root']:raise EvidenceError('assembly numerical source differs from frozen registration')
    checkpoint=read_json(prior_checkpoint)
    if public_object(prior_public)!=checkpoint:raise EvidenceError('public clean reconstruction evidence differs')
    rec=reconstruction(checkpoint,read_json(prior_observation),r)
    source=read_json(packet/'source-statement.json');verify_inventory(raw,source['archive']['inventory'])
    rebuilt=prepared_bytes(prepared,r)
    proof=read_json(retention)
    if digest(proof)!=retention_sha256:raise EvidenceError('retained replay proof differs from independent selection')
    receipts=verify_retention(proof,control,transports,job_file,job_sha256,worker_sha256)
    for receipt in receipts:complete_tree(Path(receipt['files_directory']),receipt['files'])
    by_root={item['declared_root']:receipt for item,receipt in zip(proof['roots'],receipts)}
    if replay_root not in by_root or audit_root not in by_root or replay_root==audit_root:
        raise EvidenceError('distinct complete replay and audit outputs required')
    replay_receipt=by_root[replay_root];directory=Path(replay_receipt['files_directory'])
    value=read_json(directory/'verification.json')
    audit=confined(Path(by_root[audit_root]['files_directory']),audit_relative)
    receipt,launch=process(r,audit,read_json(job_file),proof['terminal'],replay_root)
    states=replay_states(r,envelopes,chain,directory,value)
    record_files=inventory(chain,sorted(p.relative_to(chain).as_posix() for p in chain.rglob('*') if p.is_file()))
    complete_tree(chain,record_files)
    selected_public={'schema':'ovl.retained-full-replay-publication.v1','registration_sha256':digest(r),
        'job_sha256':job_sha256,'worker_sha256':worker_sha256,'retention_sha256':retention_sha256,
        'launch_sha256':digest(launch),'process_sha256':digest(receipt),'terminal_sha256':digest(proof['terminal']),
        'report_sha256':digest(value),'record_inventory_sha256':digest(record_files),
        'replay_inventory_sha256':digest(replay_receipt['files'])}
    if public_object(replay_public)!=selected_public:raise EvidenceError('public replay execution evidence differs')
    mapped=verify_replayed_exports(exports,r,value)
    mapped['scope']='both exports checked against separately retained complete replay; assembly executed no numerical updates'
    evaluation=evaluate(exports,r,prepared,{'base':value['base_model_root'],'chat':value['chat_model_root']})
    execution={'schema':'ovl.separate-computation-executions.v1',
        'mode':'earlier-complete-clean-reconstruction-and-later-continuous-full-replay',
        'reconstruction':{'report_sha256':digest(rec),'execution_observation_sha256':digest(read_json(prior_observation)),
                          'public_evidence':prior_public},
        'replay':{'report_sha256':digest(value),'audited_process_sha256':digest(receipt),'launch_sha256':digest(launch),
                  'terminal_sha256':digest(proof['terminal']),'retention_sha256':retention_sha256,'public_evidence':replay_public},
        'assembly':{'raw_inventory_sha256':digest(source['archive']['inventory']),'prepared_manifest_sha256':r['preparation_sha256'],
                    'record_inventory_sha256':digest(record_files),'replay_inventory_sha256':digest(replay_receipt['files']),
                    'safe_states_compared':states,'raw_transformations_executed':False,'numerical_updates_executed':0}}
    validate_composition(execution,r,rec,value,receipt)
    result={'schema':'ovl.complete-computation-verification.v2','result':'PASS','registration_sha256':digest(r),
        'scope':'complete-raw-transformations-fresh-initialization-all-updates-both-exported-models',
        'checks':{'publisher_ancestry':'PASS','complete_raw_reconstruction':'PASS','continuous_complete_replay':'PASS',
                  'both_export_mappings':'PASS','complete_heldout_evaluation':'PASS','fixed_greedy_inference':'PASS'},
        'endorsements':endorsements,'reconstruction':rec,'reconstructed_artifacts':rebuilt,'replay_report_sha256':digest(value),
        'replay_process':receipt,'exports':mapped,'evaluation_sha256':digest(evaluation),
        'raw_inputs':{'path_supplied_by':'caller','archive_repo':source['archive']['repo'],'archive_revision':source['archive']['revision'],
                      'archive_prefix':source['archive']['prefix'],'complete_inventory_sha256':digest(source['archive']['inventory']),
                      'all_local_bytes_rehashed':'PASS','public_anonymous_download_this_command':'NOT_RUN'},
        'base_model_root':value['base_model_root'],'chat_model_root':value['chat_model_root'],
        'reconstruction_and_input_validation_ms':checkpoint['whole_job_elapsed_ms'],
        'total_ms':(time.monotonic_ns()-started+999999)//1000000,
        'performed_by':'project-operator-separate-recorded-executions','attested_by':None,'locally_recomputed':False,
        'independent_third_party':False,'public_release_download_verification':'NOT_RUN',
        'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS','execution':execution}
    output.mkdir(parents=True);draft=output/'candidate-reports';draft.mkdir()
    for name,data in [('verification',result),('reconstruction',rec),('replay',value),('evaluation',evaluation),
                      ('exports',read_json(exports/'export.json')),('context',context)]:write_json(draft/(name+'.json'),data)
    reports(r,draft)
    if code_root()!=r['code_root']:raise EvidenceError('source changed during report assembly')
    write_json(output/'verification.json',result)
    return result


def main():
    import argparse
    from types import SimpleNamespace
    from ovl_pipeline.anchoring import PublisherPolicy
    from ovl_pipeline.production_identity import ProductionPublisherPolicy
    from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True,type=Path);p.add_argument('--selection-sha256',required=True)
    a=p.parse_args()
    try:
        value=read_json(a.selection)
        if digest(value)!=a.selection_sha256:raise EvidenceError('assembly differs from caller-selected immutable inputs')
        path_names=('packet bundle source_checkout chain progress raw prepared exports prior_checkpoint prior_observation '
                    'job_file retention output').split()
        names=('production_policy source_policy progress_policies context prior_public replay_public control transports '
               'job_sha256 worker_sha256 retention_sha256 replay_root audit_root audit_relative').split()
        fields(value,'schema '+' '.join(path_names+names),'computation assembly selection')
        if value['schema']!='ovl.computation-assembly-selection.v1':raise EvidenceError('unsupported assembly selection')
        args={k:v for k,v in value.items() if k!='schema'}
        for name in path_names:args[name]=Path(args[name])
        args['production_policy']=ProductionPublisherPolicy(**args['production_policy'])
        args['source_policy']=PublisherPolicy(**args['source_policy'])
        args['progress_policies']=[ProgressPublisherPolicy(**v) for v in args['progress_policies']]
        args['control']=SimpleNamespace(profile=args['control'])
        args['transports']=[SimpleNamespace(profile=v) for v in args['transports']]
        result=assemble(**args)
        print(canonical({'result':result['result'],'report_sha256':digest(result),
              'numerical_updates_executed_by_assembly':0,'independent_third_party':False}).decode());return 0
    except Exception as error:
        print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1


if __name__=='__main__':raise SystemExit(main())
