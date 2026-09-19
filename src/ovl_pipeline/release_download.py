"""Fresh anonymous release verification, with an explicit actual-computation mode.

Artifact verification authenticates operator claims and runs public inference. It
never turns a signed report into locally recomputed training. Full mode additionally
executes every raw transformation and training update through production_verify.full.
"""
import argparse
from pathlib import Path
import os
import re
import shutil

from huggingface_hub import HfApi,hf_hub_download
from .canonical import EvidenceError,canonical,confined,digest,file_hash,read_json,write_json
from .production_release import PAYLOAD,ReleasePublisherPolicy,validate,verify_payloads
from .anchoring import verify_anchor
from .schema import fields

ANCHOR=['release.json','release.sigstore.json']


def selection(value):
    fields(value,'base chat','fixed model selections')
    for phase,item in value.items():
        fields(item,'repo revision','fixed model selection')
        if (type(item['repo']) is not str or not re.fullmatch(r'AOSSIE/openverifiable-[a-z0-9-]+-'+phase,item['repo'])
            or type(item['revision']) is not str or not re.fullmatch('[0-9a-f]{40}',item['revision'])):
            raise EvidenceError('explicit immutable approved model selections required')


def download(selected,policy,output,*,api=None,fetch_file=None):
    """Verify the separately selected publisher before downloading model weights."""
    selection(selected)
    if type(policy) is not ReleasePublisherPolicy:raise EvidenceError('external release policy required')
    policy.validate()
    if output.exists():raise EvidenceError('release download needs a fresh output directory')
    api=api or HfApi(endpoint='https://huggingface.co',token=False);fetch_file=fetch_file or hf_hub_download
    output.mkdir(parents=True,exist_ok=False);cache=output/'transport-cache';directories={};receipts={};reference=None
    write_json(output/'intent.json',{'schema':'ovl.release-download-intent.v1','selection':selected,
        'release_sha256':policy.statement_sha256,'authentication':'anonymous-token-False','force_download':True})
    for phase in ('base','chat'):
        chosen=selected[phase];repo=chosen['repo'];revision=chosen['revision']
        try:info=api.repo_info(repo,repo_type='model',revision=revision,files_metadata=True)
        except Exception as error:raise EvidenceError('public metadata read failed: '+type(error).__name__) from None
        if info.private or info.sha!=revision:raise EvidenceError('selected public model revision unavailable')
        metadata={}
        for sibling in info.siblings:
            name=sibling.rfilename
            if name in metadata or name not in PAYLOAD+ANCHOR+['.gitattributes']:
                raise EvidenceError('extra or duplicate public model file')
            if type(sibling.size) is not int or sibling.size<0:raise EvidenceError('missing public model file size')
            metadata[name]=sibling.size
        if sorted(n for n in metadata if n!='.gitattributes')!=sorted(PAYLOAD+ANCHOR):
            raise EvidenceError('incomplete closed public model file set')
        target=output/phase;target.mkdir();directories[phase]=target
        def get(name,maximum,expected=None):
            if metadata[name]>maximum:raise EvidenceError('public model metadata exceeds bounded size')
            try:
                raw=Path(fetch_file(repo_id=repo,repo_type='model',revision=revision,filename=name,token=False,
                    force_download=True,local_files_only=False,cache_dir=cache)).resolve(strict=True)
            except Exception as error:raise EvidenceError('public model download failed: '+type(error).__name__) from None
            if not raw.is_relative_to(cache.resolve()) or not raw.is_file():raise EvidenceError('model transport escaped fresh cache')
            size=raw.stat().st_size
            if size!=metadata[name] or size>maximum:raise EvidenceError('downloaded model file size differs')
            actual={'path':name,'bytes':size,'sha256':file_hash(raw)}
            if expected is not None and actual!=expected:raise EvidenceError('downloaded model bytes differ from signed inventory')
            path=confined(target,name);path.parent.mkdir(parents=True,exist_ok=True);os.link(raw,path,follow_symlinks=False)
            with path.open('rb') as f:os.fsync(f.fileno())
            return actual
        files=[get('release.json',1024**2),get('release.sigstore.json',4*1024**2)]
        identity=verify_anchor(target/'release.json',target/'release.sigstore.json',policy)
        value=validate(read_json(target/'release.json'))
        if digest(value)!=policy.statement_sha256:raise EvidenceError('release differs from external selection')
        if reference is None:reference=value
        elif value!=reference or file_hash(target/'release.sigstore.json')!=file_hash(directories['base']/'release.sigstore.json'):
            raise EvidenceError('model repositories carry different release evidence')
        if any(selected[p]['repo']!=value['models'][p]['repo'] for p in ('base','chat')):
            raise EvidenceError('release names differ from independently selected destinations')
        for entry in value['models'][phase]['files']:files.append(get(entry['path'],entry['bytes'],entry))
        if '.gitattributes' in metadata:files.append(get('.gitattributes',65536))
        receipts[phase]={'repo':repo,'revision':revision,'files':files,'publisher':identity,
                         'host_generated_attributes':'downloaded-and-recorded-not-part-of-signed-model-payload' if '.gitattributes' in metadata else 'absent'}
    result={'schema':'ovl.public-model-byte-download.v1','result':'PASS','release_sha256':digest(reference),
            'models':receipts,'authentication':'anonymous-token-False','force_download':True,
            'scope':'all-selected-public-file-bytes-and-external-publisher-identity',
            'locally_recomputed_training':False,'training_replay':'NOT_RUN'}
    write_json(output/'verification.json',result);return reference,directories,result


def inference(directories,r,value,evidence):
    from .production_export import infer
    from .production_verify import PUBLIC_PROMPTS
    evaluation=read_json(evidence/'evaluation.json');results={};outcomes=[]
    for phase in ('base','chat'):
        demos=read_json(directories[phase]/'demonstrations.json')
        if demos!=evaluation['models'][phase]['demonstrations'] or [d['prompt'] for d in demos]!=PUBLIC_PROMPTS:
            raise EvidenceError('downloaded public demonstrations differ from full evaluation')
        observations=[]
        for recorded in demos:
            actual=infer(directories[phase]/'model',r,phase,recorded['prompt'],max_new_tokens=32,
                         expected_model_root=value['models'][phase]['model_root'])
            repeated=infer(directories[phase]/'model',r,phase,recorded['prompt'],max_new_tokens=32,
                           expected_model_root=value['models'][phase]['model_root'])
            if actual!=repeated:raise EvidenceError('downloaded model inference is not repeatable on this runtime')
            semantic=lambda x:{k:v for k,v in x.items() if k not in ('runtime','performed_by')}
            runtime_equal=actual['runtime']==recorded['runtime'];same_output=semantic(actual)==semantic(recorded)
            outcome='PASS' if same_output else ('FAIL' if runtime_equal else 'UNSUPPORTED')
            outcomes.append(outcome)
            observations.append({'actual':actual,'recorded':recorded,'repeated_generation':'PASS','result':outcome,
                                 'runtime_matches_operator_observation':runtime_equal,
                                 'performer_labels':{'recorded':recorded['performed_by'],'observed':actual['performed_by']}})
        results[phase]=observations
    overall='FAIL' if 'FAIL' in outcomes else ('UNSUPPORTED' if 'UNSUPPORTED' in outcomes else 'PASS')
    return {'schema':'ovl.downloaded-model-inference.v1','result':overall,'models':results,
            'scope':'fixed-public-prompts-greedy-output-token-identity-and-local-repeatability',
            'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'}


def verify(selected,release_policy,production_policy,source_policy,progress_policies,source_checkout,output,*,
           mode='artifacts',raw=None,runtime=None):
    from .release_commitment import verify_parents
    from .progress_commitment import download_archive
    if mode not in ('artifacts','full'):raise EvidenceError('explicit artifact or complete computation mode required')
    if mode=='full' and (raw is None or runtime is None):raise EvidenceError('full mode requires raw archive and audited replay runtime')
    if mode=='artifacts' and (raw is not None or runtime is not None):raise EvidenceError('computation inputs require full mode')
    if output.exists():raise EvidenceError('release verification requires fresh output')
    output.mkdir(parents=True,exist_ok=False)
    value,directories,transport=download(selected,release_policy,output/'models')
    evidence=output/'evidence';archive=download_archive(value['evidence'],evidence)
    r,ancestry=verify_parents(value,evidence,output/'parents',source_checkout,production_policy,source_policy,
                            progress_policies,complete_checkpoints=mode=='full')
    payload=verify_payloads(directories['base']/'release.json',directories['base']/'release.sigstore.json',release_policy,directories,r)
    computed=None
    if mode=='full':
        from .production_verify import full
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
    write_json(output/'verification.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['artifacts','full'])
    for n in ('selection','release-policy','production-policy','source-policy','progress-policies','source-checkout','output'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('raw','lock','wheels','venv','source','interpreter-archive','interpreter-root','allowed-generated'):
        p.add_argument('--'+n,type=Path)
    p.add_argument('--interpreter-sha256');a=p.parse_args()
    from .anchoring import PublisherPolicy
    from .production_identity import ProductionPublisherPolicy
    from .progress_anchoring import ProgressPublisherPolicy
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
