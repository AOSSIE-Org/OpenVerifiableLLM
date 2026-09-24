"""Fresh anonymous release downloads plus actual historical reconstruction/replay.

Use the independently selected immutable driver revision. Artifact-only checking
remains available through the frozen ovl_pipeline.release_download command.
"""
import argparse
from pathlib import Path
import shutil
from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,write_json
from ovl_pipeline.production_release import ReleasePublisherPolicy,verify_payloads
from ovl_pipeline.release_download import download,inference
from historical_reconstruction import driver_identity,admit_runtime_source


def verify(selected,release_policy,production_policy,source_policy,progress_policies,source_checkout,output,*,
           mode='full',raw=None,runtime=None):
    from ovl_pipeline.release_commitment import verify_parents
    from ovl_pipeline.progress_commitment import download_archive
    if mode!='full':raise EvidenceError('this driver requires full computation; use the frozen artifact command for artifact verification')
    if mode=='full' and (raw is None or runtime is None):raise EvidenceError('full mode requires raw archive and audited replay runtime')
    if mode=='artifacts' and (raw is not None or runtime is not None):raise EvidenceError('computation inputs require full mode')
    identity=driver_identity(source_checkout,entrypoint=__file__)
    admit_runtime_source(source_checkout,runtime)
    if output.exists():raise EvidenceError('release verification requires fresh output')
    output.mkdir(parents=True,exist_ok=False)
    value,directories,transport=download(selected,release_policy,output/'models')
    evidence=output/'evidence';archive=download_archive(value['evidence'],evidence)
    r,ancestry=verify_parents(value,evidence,output/'parents',source_checkout,production_policy,source_policy,
                            progress_policies,complete_checkpoints=mode=='full')
    payload=verify_payloads(directories['base']/'release.json',directories['base']/'release.sigstore.json',release_policy,directories,r)
    computed=None
    if mode=='full':
        from verify_complete import full
        exports=output/'exports';exports.mkdir()
        shutil.copyfile(evidence/'exports.json',exports/'export.json')
        for phase in ('base','chat'):shutil.copytree(directories[phase]/'model',exports/phase)
        parents=output/'parents'
        computed=full(parents/'packet',parents/'registration-anchor/registration.sigstore.json',production_policy,source_policy,
            source_checkout,parents/'chain',parents/'progress',progress_policies,raw,exports,output/'computation',runtime)
        if (computed['result']!='PASS' or computed['locally_recomputed'] is not True
            or computed['base_model_root']!=value['models']['base']['model_root']
            or computed['chat_model_root']!=value['models']['chat']['model_root']):raise EvidenceError('full downloaded-release computation did not pass')
    generated=inference(directories,r,value,evidence);write_json(output/'inference.json',generated)
    report={'schema':'ovl.public-release-verification.v1','result':generated['result'],'mode':mode,'release_sha256':digest(value),
        'downloads':transport,'evidence_download':archive,'ancestry':ancestry,'payloads':payload,'inference_sha256':digest(generated),'inference_result':generated['result'],
        'scope':'complete-public-download-identity-inference'+('-and-fresh-full-reconstruction-replay' if computed else '-and-operator-attestation'),
        'locally_recomputed_training':computed is not None,'complete_computation_sha256':digest(computed) if computed else None,
        'raw_reconstruction':'PASS' if computed else 'NOT_RUN','continuous_replay':'PASS' if computed else 'NOT_RUN',
        'performed_by':'verifier-operator','independent_third_party':False,
        'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'}
    if driver_identity(source_checkout,entrypoint=__file__)!=identity:raise EvidenceError('release verifier driver changed')
    write_json(output/'verification.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['full'])
    for n in ('selection','release-policy','production-policy','source-policy','progress-policies','source-checkout','output'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('raw','lock','wheels','venv','source','interpreter-archive','interpreter-root','allowed-generated'):
        p.add_argument('--'+n,type=Path)
    p.add_argument('--interpreter-sha256');a=p.parse_args()
    from ovl_pipeline.anchoring import PublisherPolicy
    from ovl_pipeline.production_identity import ProductionPublisherPolicy
    from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
    try:
        keys=('lock','wheels','venv','source','interpreter_archive','interpreter_sha256','interpreter_root')
        runtime=None
        if a.mode=='full':
            runtime={k:getattr(a,k) for k in keys}
            if any(v is None for v in runtime.values()):raise EvidenceError('full mode requires every public runtime selection')
            runtime['allowed_generated']=read_json(a.allowed_generated) if a.allowed_generated else {}
        elif any(getattr(a,k) is not None for k in (*keys,'allowed_generated')):raise EvidenceError('runtime options require full mode')
        result=verify(read_json(a.selection),ReleasePublisherPolicy(**read_json(a.release_policy)),
            ProductionPublisherPolicy(**read_json(a.production_policy)),PublisherPolicy(**read_json(a.source_policy)),
            [ProgressPublisherPolicy(**v) for v in read_json(a.progress_policies)],a.source_checkout,a.output,
            mode=a.mode,raw=a.raw,runtime=runtime)
        print(canonical({'result':result['result'],'scope':result['scope'],'report_sha256':digest(result)}).decode());return 0 if result['result']=='PASS' else (2 if result['result']=='UNSUPPORTED' else 1)
    except Exception as error:print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1

if __name__=='__main__':raise SystemExit(main())
