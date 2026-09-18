"""Complete raw reconstruction and continuous replay, never a saved-PASS shortcut.

The coordinator runs in a separately trusted CPU preparation/verifier environment.
It reconstructs all transformations afresh, then starts a newly audited compatible
GPU process to recompute every update. A public-release download/identity check is
separate; neither that check nor these results establish factual answer accuracy.
"""
import argparse
from pathlib import Path
import struct
import time

from .canonical import EvidenceError,canonical,confined,digest,read_json,verify_inventory,write_json
from .schema import fields

STAGES=['corpus','tokenizer','wikipedia','conversation-selection','conversation','conversation-validation']
PUBLIC_PROMPTS=['What is an encyclopedia?','Explain how rain forms.']


def evaluate(exports,r,prepared,roots):
    """Every held-out conversation target, fixed CPU FP32 loss arithmetic."""
    import torch
    from torch.nn import functional as F
    from .data import batches,check_coverage,validate_stream
    from .production_export import load_model,infer
    from .training import environment
    from .gpu import host_runtime
    manifest=read_json(prepared/'preparation.json')
    if digest(manifest)!=r['preparation_sha256'] or manifest['validation_used_for_training'] is not False:
        raise EvidenceError('evaluation requires the registered held-out preparation')
    directory=prepared/'conversation-validation';stream=read_json(directory/'stream.json')
    if stream!=manifest['streams']['conversation-validation']:raise EvidenceError('held-out stream differs from preparation')
    validate_stream(directory,stream)
    reports={}
    for phase in ('base','chat'):
        model=load_model(exports/phase,r,phase,expected_model_root=roots[phase]);cursor=0;total_loss=0.0;count=0
        with torch.no_grad():
            for batch in batches(directory,r['recipe']['context'],1):
                following=check_coverage(batch,cursor,stream['targets'])
                logits=model(batch['inputs'])
                losses=F.cross_entropy(logits.flatten(0,1),batch['targets'].flatten(),reduction='none')
                nll=losses[batch['mask'].flatten()].sum(dtype=torch.float64).item()
                if not torch.isfinite(torch.tensor(nll,dtype=torch.float64)):raise EvidenceError('nonfinite validation loss')
                total_loss+=nll;cursor=following;count+=1
        if cursor!=stream['targets'] or cursor<=0:raise EvidenceError('held-out evaluation coverage incomplete')
        demonstrations=[]
        for prompt in PUBLIC_PROMPTS:
            first=infer(exports/phase,r,phase,prompt,max_new_tokens=32,expected_model_root=roots[phase])
            second=infer(exports/phase,r,phase,prompt,max_new_tokens=32,expected_model_root=roots[phase])
            if first!=second:raise EvidenceError('fixed greedy inference was not reproducible on this runtime')
            demonstrations.append(first)
        reports[phase]={'model_root':roots[phase],'targets':cursor,'batches':count,
                       'negative_log_likelihood_sum_float64_hex':struct.pack('>d',total_loss).hex(),
                       'mean_negative_log_likelihood_float64_hex':struct.pack('>d',total_loss/cursor).hex(),
                       'demonstrations':demonstrations}
    return {'schema':'ovl.complete-heldout-evaluation.v1','registration_sha256':digest(r),
            'split':'official-conversation-validation','stream_sha256':digest(stream),'subset_sampling':False,
            'recipe':{'device':'cpu','dtype':'float32','batch_size':1,'context':r['recipe']['context'],
                      'loss':'masked-FP32-cross-entropy-FP64-batch-sum-sequential-FP64-total-v1'},
            'runtime':{'software':environment(),'host':host_runtime()},'models':reports,
            'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS','independent_third_party':False}


def full(packet,bundle,production_policy,source_policy,source_checkout,chain_directory,progress_directory,
         progress_policies,raw,expected_exports,output,runtime):
    """Fresh work is mandatory: no argument accepts prior reconstruction/replay."""
    from dataclasses import asdict
    from .production_replay import authenticate
    from .production_anchoring import object_at
    from .preparation import prepare_committed
    from .prepared_verification import verify_prepared
    from .production_export import verify_replayed_exports
    from .runtime_launch import launch
    from .training import code_root
    if output.exists():raise EvidenceError('full verifier requires fresh output; preserve interrupted attempts')
    fields(runtime,'lock wheels venv source allowed_generated interpreter_archive interpreter_sha256 interpreter_root','full verifier runtime')
    if runtime['interpreter_archive'] is None or runtime['interpreter_sha256'] is None or runtime['interpreter_root'] is None:
        raise EvidenceError('full replay requires explicitly selected public interpreter origin')
    started=time.monotonic_ns()
    r,envelopes,endorsements=authenticate(packet,bundle,production_policy,source_policy,source_checkout,
                                         chain_directory,progress_directory,progress_policies)
    if r['code_root']!=code_root():raise EvidenceError('full verifier source differs from registration')
    source=object_at(packet,'source-statement.json');prepared=object_at(packet,'preparation.json')
    if digest(prepared)!=r['preparation_sha256'] or digest(source)!=r['source_statement_sha256']:
        raise EvidenceError('full verifier data parents differ from registration')
    # Rehash every archived raw input and receipt. prepare_committed additionally
    # performs complete source decompression and all transformations from scratch.
    verify_inventory(raw,source['archive']['inventory'])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'selected-policies.json',{'production':asdict(production_policy),'source':asdict(source_policy),
                                              'progress':[asdict(p) for p in progress_policies]})
    write_json(output/'production-policy.json',asdict(production_policy));write_json(output/'source-policy.json',asdict(source_policy))
    write_json(output/'progress-policies.json',[asdict(p) for p in progress_policies])
    reconstruction=prepare_committed(packet/'source-statement.json',packet/'source-statement.sigstore.json',source_policy,
        raw/'wikipedia',raw/'conversation',output/'reconstructed',expected_preparation=prepared,resume=False)
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
    write_json(output/'verification.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('packet','registration-bundle','production-policy','source-policy','source-checkout','chain-directory',
              'progress-directory','progress-policies','raw','exports','output','lock','wheels','venv','source',
              'interpreter-archive','interpreter-root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--interpreter-sha256',required=True);p.add_argument('--allowed-generated',type=Path);a=p.parse_args()
    from .anchoring import PublisherPolicy
    from .production_identity import ProductionPublisherPolicy
    from .progress_anchoring import ProgressPublisherPolicy
    try:
        runtime={k:getattr(a,k) for k in ('lock','wheels','venv','source','interpreter_archive','interpreter_sha256','interpreter_root')}
        runtime['allowed_generated']=read_json(a.allowed_generated) if a.allowed_generated else {}
        result=full(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
                    PublisherPolicy(**read_json(a.source_policy)),a.source_checkout,a.chain_directory,a.progress_directory,
                    [ProgressPublisherPolicy(**v) for v in read_json(a.progress_policies)],a.raw,a.exports,a.output,runtime)
        print(canonical({'result':result['result'],'scope':result['scope'],'report_sha256':digest(result)}).decode());return 0
    except Exception as error:print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1

if __name__=='__main__':raise SystemExit(main())
