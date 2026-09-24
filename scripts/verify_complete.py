"""Fresh verification using a separately pinned driver and frozen numerical code.

The historical preparation process runs the exact signed source inventory. No
saved result or prepared cache may replace fresh transformations or full replay.
Select this script's immutable revision independently; keep the registered module
tree and requirements/gpu.lock unchanged. Driver identity is recorded separately.
"""
import argparse
from pathlib import Path
import time

from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,verify_inventory,write_json
from ovl_pipeline.schema import fields
from ovl_pipeline.production_verify import STAGES,evaluate
from historical_reconstruction import reconstruct,driver_identity,check_replay_launch,admit_runtime_source


def full(packet,bundle,production_policy,source_policy,source_checkout,chain_directory,progress_directory,
         progress_policies,raw,expected_exports,output,runtime):
    """Fresh work is mandatory: no argument accepts prior reconstruction/replay."""
    from dataclasses import asdict
    from ovl_pipeline.production_replay import authenticate
    from ovl_pipeline.production_anchoring import object_at
    from ovl_pipeline.prepared_verification import verify_prepared
    from ovl_pipeline.production_export import verify_replayed_exports
    from ovl_pipeline.runtime_launch import launch
    from ovl_pipeline.training import code_root
    if output.exists():raise EvidenceError('full verifier requires fresh output; preserve interrupted attempts')
    fields(runtime,'lock wheels venv source allowed_generated interpreter_archive interpreter_sha256 interpreter_root','full verifier runtime')
    if runtime['interpreter_archive'] is None or runtime['interpreter_sha256'] is None or runtime['interpreter_root'] is None:
        raise EvidenceError('full replay requires explicitly selected public interpreter origin')
    identity=driver_identity(source_checkout,entrypoint=__file__)
    started=time.monotonic_ns()
    r,envelopes,endorsements=authenticate(packet,bundle,production_policy,source_policy,source_checkout,
                                         chain_directory,progress_directory,progress_policies)
    if r['code_root']!=code_root():raise EvidenceError('full verifier source differs from registration')
    source=object_at(packet,'source-statement.json');prepared=object_at(packet,'preparation.json')
    if digest(prepared)!=r['preparation_sha256'] or digest(source)!=r['source_statement_sha256']:
        raise EvidenceError('full verifier data parents differ from registration')
    admit_runtime_source(source_checkout,runtime)
    # Rehash every archived raw input and receipt. prepare_committed additionally
    # performs complete source decompression and all transformations from scratch.
    verify_inventory(raw,source['archive']['inventory'])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'driver.json',identity)
    write_json(output/'selected-policies.json',{'production':asdict(production_policy),'source':asdict(source_policy),
                                              'progress':[asdict(p) for p in progress_policies]})
    write_json(output/'production-policy.json',asdict(production_policy));write_json(output/'source-policy.json',asdict(source_policy))
    write_json(output/'progress-policies.json',[asdict(p) for p in progress_policies])
    reconstruction=reconstruct(source_checkout,packet/'source-statement.json',packet/'source-statement.sigstore.json',
        source_policy,raw,output/'reconstructed',prepared,output/'historical-preparation')
    if (reconstruction.get('result')!='PASS' or reconstruction.get('full_reconstruction_compared') is not True
        or reconstruction.get('preparation_sha256')!=r['preparation_sha256']
        or reconstruction.get('stages_executed_this_run')!=STAGES or reconstruction.get('stages_adopted_from_local_cache')!=[]):
        raise EvidenceError('incomplete or cached reconstruction cannot pass the full profile')
    write_json(output/'reconstruction.json',reconstruction)
    rebuilt=verify_prepared(output/'reconstructed',r['preparation_sha256'],r['source_statement_sha256'])
    reconstruction_ms=(time.monotonic_ns()-started+999999)//1000000
    if code_root()!=r['code_root']:raise EvidenceError('verifier source changed during reconstruction')
    arguments=[]
    paths={'packet':packet,'registration-bundle':bundle,'production-policy':output/'production-policy.json',
           'source-policy':output/'source-policy.json','source-checkout':source_checkout,'chain-directory':chain_directory,
           'progress-directory':progress_directory,'progress-policies':output/'progress-policies.json',
           'wikipedia-stream':output/'reconstructed/wikipedia','conversation-stream':output/'reconstructed/conversation',
           'exports':expected_exports,'output':output/'numerical-replay'}
    for name,path in paths.items():arguments.extend(['--'+name,str(path.resolve())])
    # This call executes the actual child. It does not take a replay report from
    # the caller. The trusted parent audited every target dependency first.
    process=launch(runtime['lock'],runtime['wheels'],runtime['venv'],runtime['source'],output/'gpu-launch',
                   'ovl_pipeline.production_export',['replay-check',*arguments],
                   allowed_generated=runtime['allowed_generated'],interpreter_archive=runtime['interpreter_archive'],
                   interpreter_sha256=runtime['interpreter_sha256'],interpreter_root=runtime['interpreter_root'])
    if process['exit_code']!=0:raise EvidenceError('full replay process failed; saved reports cannot override process exit')
    replay=read_json(output/'numerical-replay/verification.json')
    check_replay_launch(output,process,r,envelopes,runtime,arguments,replay)
    exports=verify_replayed_exports(expected_exports,r,replay)
    if code_root()!=r['code_root']:raise EvidenceError('full replay process or source changed')
    evaluation=evaluate(expected_exports,r,output/'reconstructed',{'base':replay['base_model_root'],'chat':replay['chat_model_root']})
    write_json(output/'evaluation.json',evaluation)
    if exports.get('result')!='PASS':raise EvidenceError('both replayed export mappings must pass')
    report={'schema':'ovl.complete-computation-verification.v1','result':'PASS','registration_sha256':digest(r),
            'scope':'complete-raw-transformations-fresh-initialization-all-updates-both-exported-models',
            'checks':{'publisher_ancestry':'PASS','complete_raw_reconstruction':reconstruction['result'],'continuous_complete_replay':replay['result'],
                      'both_export_mappings':exports['result'],'complete_heldout_evaluation':'PASS','fixed_greedy_inference':'PASS'},
            'endorsements':endorsements,'reconstruction':reconstruction,'reconstructed_artifacts':rebuilt,
            'replay_report_sha256':digest(replay),'replay_process':process,'exports':exports,'evaluation_sha256':digest(evaluation),
            'raw_inputs':{'path_supplied_by':'caller','archive_repo':source['archive']['repo'],'archive_revision':source['archive']['revision'],
                          'archive_prefix':source['archive']['prefix'],'complete_inventory_sha256':digest(source['archive']['inventory']),
                          'all_local_bytes_rehashed':'PASS','public_anonymous_download_this_command':'NOT_RUN'},
            'base_model_root':replay['base_model_root'],'chat_model_root':replay['chat_model_root'],
            'reconstruction_and_input_validation_ms':reconstruction_ms,'total_ms':(time.monotonic_ns()-started+999999)//1000000,
            'performed_by':'verifier-operator','attested_by':None,'locally_recomputed':True,'independent_third_party':False,
            'public_release_download_verification':'NOT_RUN','factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'}
    if driver_identity(source_checkout,entrypoint=__file__)!=identity:raise EvidenceError('verification driver changed during execution')
    write_json(output/'verification.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('packet','registration-bundle','production-policy','source-policy','source-checkout','chain-directory',
              'progress-directory','progress-policies','raw','exports','output','lock','wheels','venv','source',
              'interpreter-archive','interpreter-root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--interpreter-sha256',required=True);p.add_argument('--allowed-generated',type=Path);a=p.parse_args()
    from ovl_pipeline.anchoring import PublisherPolicy
    from ovl_pipeline.production_identity import ProductionPublisherPolicy
    from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
    try:
        runtime={k:getattr(a,k) for k in ('lock','wheels','venv','source','interpreter_archive','interpreter_sha256','interpreter_root')}
        runtime['allowed_generated']=read_json(a.allowed_generated) if a.allowed_generated else {}
        result=full(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
                    PublisherPolicy(**read_json(a.source_policy)),a.source_checkout,a.chain_directory,a.progress_directory,
                    [ProgressPublisherPolicy(**v) for v in read_json(a.progress_policies)],a.raw,a.exports,a.output,runtime)
        print(canonical({'result':result['result'],'scope':result['scope'],'report_sha256':digest(result)}).decode());return 0
    except Exception as error:print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1

if __name__=='__main__':raise SystemExit(main())
