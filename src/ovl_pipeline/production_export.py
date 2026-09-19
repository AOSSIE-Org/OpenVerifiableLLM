"""Safe production candidate exports and externally selected local inference.

Exporting an authenticated checkpoint does not prove its training trajectory. The
full verifier must compare these weights against its freshly replayed model roots.
Neither provenance nor reproducible generation establishes factual accuracy.
"""
import argparse
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file,save,save_file
from tokenizers import Tokenizer

from .canonical import EvidenceError,confined,digest,file_hash,inventory,read_json,sha256,verify_inventory,write_json
from .data import ASSISTANT,BOS,EOS,OFFSET,PAD,USER,text_ids
from .production_contract import validate
from .production_replay import authenticate
from .schema import fields,integer
from .state import aliases,read_state,unpack,tensor_digest
from .training import code_root,initialize

FILES=['config.json','model.safetensors','preparation.json','tokenizer-manifest.json','tokenizer.json']
CONTROLS={'pad':PAD,'bos':BOS,'eos':EOS,'user':USER,'assistant':ASSISTANT}


def model_for_inference(registration):
    validate(registration)
    # Batch size is a training allocation limit, not an architecture parameter.
    # Construct the identical architecture with a one-sequence CPU allocation.
    recipe={**registration['recipe'],'batch_size':1}
    model,opt,_=initialize(recipe)
    return model,opt


def configuration(r,phase,root,tokenizer):
    if phase not in ('base','chat'):raise EvidenceError('explicit base/chat phase required')
    return {'schema':'ovl.production-export-config.v1','registration_sha256':digest(r),'phase':phase,
            'recipe':r['recipe'],'loader_code_root':r['code_root'],'model_root':root,
            'aliases':{'lm_head.weight':'transformer.wte.weight'},
            'tokenizer':{'file':'tokenizer.json','sha256':file_hash(tokenizer/'tokenizer.json'),
                         'manifest_file':'tokenizer-manifest.json','manifest_sha256':file_hash(tokenizer/'tokenizer-manifest.json'),
                         'text_id_offset':OFFSET,'controls':CONTROLS},
            'inference':{'input_format':'bos-text-v1' if phase=='base' else 'bos-user-text-eos-assistant-v1',
                         'decoding':'greedy','default_max_new_tokens':64,'maximum_new_tokens':256,
                         'context_policy':'retain-last-context-tokens-v1','device':'cpu','parameter_dtype':'float32'}}


def check_tokenizer(r,path,prepared_manifest):
    root=file_hash(confined(path,'tokenizer.json'))
    if digest(prepared_manifest)!=r['preparation_sha256']:raise EvidenceError('tokenizer preparation parent differs from registration')
    m=read_json(confined(path,'tokenizer-manifest.json'))
    if (m!=prepared_manifest['tokenizer'] or m.get('text_id_offset')!=OFFSET or m.get('controls')!=CONTROLS
        or m.get('vocab_size')!=r['recipe']['model']['vocab_size'] or [e['path'] for e in m['files']]!=['tokenizer.json']):
        raise EvidenceError('tokenizer recipe differs from registered preparation')
    verify_inventory(path,m['files'])
    return root


def checked_weights(model,state):
    expected=dict(model.state_dict())
    if set(state)!=set(expected) or any(state[k].dtype!=expected[k].dtype or state[k].shape!=expected[k].shape for k in state):
        raise EvidenceError('model tensor names/dtypes/shapes differ from architecture')
    for alias,primary in aliases(model).items():
        if not torch.equal(state[alias],state[primary]):raise EvidenceError('inconsistent tied model tensor')
    if any(not torch.isfinite(t).all() for t in state.values()):raise EvidenceError('nonfinite exported model')


def export_candidates(packet,bundle,production_policy,source_policy,source_checkout,chain_directory,
                      progress_directory,progress_policies,prepared_directory,output):
    if output.exists():raise EvidenceError('candidate export output must be fresh')
    r,envelopes,endorsements=authenticate(packet,bundle,production_policy,source_policy,source_checkout,
                                         chain_directory,progress_directory,progress_policies)
    if r['code_root']!=code_root():raise EvidenceError('export source differs from registered code')
    from .prepared_verification import verify_prepared
    prepared=verify_prepared(prepared_directory,r['preparation_sha256'],r['source_statement_sha256'])
    # The actual prepared-manifest bytes, not a receipt's claimed PASS, are the
    # registration parent. verify_prepared independently rechecks all stage bytes.
    if file_hash(prepared_directory/'preparation.json')!=r['preparation_sha256']:
        raise EvidenceError('export prepared inputs differ from registration')
    prepared_manifest=read_json(prepared_directory/'preparation.json')
    tokenizer=prepared_directory/'tokenizer';check_tokenizer(r,tokenizer,prepared_manifest)
    output.mkdir(parents=True,exist_ok=False);exports={}
    for phase,kind in [('base','base'),('chat','final')]:
        selected=[e for e in envelopes if e['body']['kind']==kind]
        if len(selected)!=1:raise EvidenceError('missing/ambiguous final phase boundary')
        envelope=selected[0];b=envelope['body'];model,_=model_for_inference(r)
        md,tensors=read_state(confined(chain_directory,b['checkpoint_path']),b['checkpoint'])
        obj=unpack(md['tree'],tensors)
        if obj['control']!=b['control'] or obj['aliases']!=aliases(model):raise EvidenceError('checkpoint control/aliases differ')
        state=obj['model'];checked_weights(model,state);root=tensor_digest(state)
        del state['lm_head.weight'];dest=output/phase;dest.mkdir()
        save_file({k:v.contiguous().clone() for k,v in state.items()},str(dest/'model.safetensors'))
        for name in ('tokenizer.json','tokenizer-manifest.json'):shutil.copyfile(tokenizer/name,dest/name)
        shutil.copyfile(prepared_directory/'preparation.json',dest/'preparation.json')
        config=configuration(r,phase,root,tokenizer);write_json(dest/'config.json',config)
        exports[phase]={'phase':phase,'boundary_sha256':digest(envelope),'checkpoint_state_root':b['checkpoint']['state_root'],
                        'model_root':root,'files':inventory(dest,FILES)}
    report={'schema':'ovl.production-candidate-export.v1','status':'EXPORTED_NOT_TRAINING_VERIFIED',
            'registration_sha256':digest(r),'chain_sha256':digest(envelopes),'exports':exports,
            'endorsements':endorsements,'prepared_artifact_check':prepared,
            'full_raw_reconstruction':'NOT_RUN','continuous_numerical_replay':'NOT_RUN','public_download_verification':'NOT_RUN',
            'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE'}
    write_json(output/'export.json',report);return report


def load_model(directory,r,phase,*,expected_model_root=None):
    if r['code_root']!=code_root():raise EvidenceError('trusted loader differs from registered source')
    config=read_json(confined(directory,'config.json'))
    fields(config,'schema registration_sha256 phase recipe loader_code_root model_root aliases tokenizer inference','production export configuration')
    expected=configuration(r,phase,config['model_root'],directory)
    if config!=expected:raise EvidenceError('export configuration differs from externally selected registration/inference recipe')
    check_tokenizer(r,directory,read_json(confined(directory,'preparation.json')))
    if expected_model_root is not None and config['model_root']!=expected_model_root:
        raise EvidenceError('export model root differs from freshly replayed model')
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()) or sorted(p.name for p in directory.iterdir())!=FILES:
        raise EvidenceError('unexpected candidate model payload files')
    path=confined(directory,'model.safetensors')
    if path.stat().st_size>512*1024**2:raise EvidenceError('export exceeds bounded model allocation')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        if f.metadata():raise EvidenceError('unexpected model tensor metadata')
    model,_=model_for_inference(r);expected_state=dict(model.state_dict());del expected_state['lm_head.weight']
    state=load_file(str(path))
    if set(state)!=set(expected_state) or any(state[k].dtype!=expected_state[k].dtype or state[k].shape!=expected_state[k].shape for k in state):
        raise EvidenceError('export tensor names/dtypes/shapes differ from architecture')
    if sha256(save(state))!=file_hash(path):raise EvidenceError('noncanonical model tensor serialization')
    state['lm_head.weight']=state['transformer.wte.weight'];checked_weights(model,state);model.load_state_dict(state,strict=True)
    if tensor_digest(dict(model.state_dict()))!=config['model_root']:raise EvidenceError('exported weight root differs')
    model.eval();return model


def infer(directory,r,phase,prompt,*,max_new_tokens=64,expected_model_root=None):
    integer(max_new_tokens,1,256,'generated token bound')
    if type(prompt) is not str or len(prompt.encode())>65536:raise EvidenceError('prompt exceeds bounded UTF8 input')
    model=load_model(directory,r,phase,expected_model_root=expected_model_root)
    tok=Tokenizer.from_file(str(confined(directory,'tokenizer.json')))
    tokens=[BOS]+text_ids(tok,prompt) if phase=='base' else [BOS,USER]+text_ids(tok,prompt)+[EOS,ASSISTANT]
    inputs=tokens.copy();generated=[];context=r['recipe']['context']
    with torch.no_grad():
        for _ in range(max_new_tokens):
            value=int(model(torch.tensor([tokens[-context:]],dtype=torch.int64))[0,-1].argmax())
            tokens.append(value);generated.append(value)
            if value==EOS:break
    from .training import environment
    from .gpu import host_runtime
    return {'schema':'ovl.local-generation.v1','registration_sha256':digest(r),'phase':phase,'prompt':prompt,
            'prompt_utf8_sha256':sha256(prompt.encode()),'model_root':read_json(directory/'config.json')['model_root'],
            'input_ids':inputs,'output_ids':generated,'decoded_text':tok.decode([v-OFFSET for v in generated if v>=OFFSET]),
            'control_output_ids':[v for v in generated if v<OFFSET],'decoding':'greedy','max_new_tokens':max_new_tokens,
            'sampling_rng_used':False,'runtime':{'software':environment(),'host':host_runtime()},
            'inference_config_sha256':file_hash(directory/'config.json'),'performed_by':'local-execution-operator',
            'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE'}


def verify_replayed_exports(directory,r,replay_result):
    """Consume the in-process full replay result; saved assertions are not replay."""
    if (replay_result.get('schema')!='ovl.production-numerical-replay.v1' or replay_result.get('result')!='PASS'
        or replay_result.get('registration_sha256')!=digest(r) or replay_result.get('initial_state_regenerated') is not True
        or replay_result.get('prover_checkpoints_restored') is not False):raise EvidenceError('fresh complete replay result required')
    contract=validate(r)
    if (replay_result.get('updates_recomputed')!={p:r['coverage'][p]['updates'] for p in ('wikipedia','conversation')}
        or replay_result.get('targets_recomputed')!={p:r['coverage'][p]['targets'] for p in ('wikipedia','conversation')}
        or len(replay_result.get('comparisons',[]))!=contract['primary_boundaries']):
        raise EvidenceError('replay coverage/comparisons are incomplete')
    report=read_json(confined(directory,'export.json'))
    fields(report,'schema status registration_sha256 chain_sha256 exports endorsements prepared_artifact_check full_raw_reconstruction continuous_numerical_replay public_download_verification factual_accuracy','candidate export report')
    if (report['schema']!='ovl.production-candidate-export.v1' or report['status']!='EXPORTED_NOT_TRAINING_VERIFIED'
        or set(report['exports'])!={'base','chat'}):
        raise EvidenceError('unsupported candidate export report')
    if report['registration_sha256']!=digest(r) or report['chain_sha256']!=replay_result['chain_sha256']:
        raise EvidenceError('candidate export ancestry differs from replay')
    results={}
    from .production_chain import schedule
    expected=schedule(r)
    for phase in ('base','chat'):
        selected=report['exports'][phase]
        fields(selected,'phase boundary_sha256 checkpoint_state_root model_root files','phase export')
        boundary=next(x for x in expected if x['kind']==('base' if phase=='base' else 'final'))
        comparison=replay_result['comparisons'][boundary['index']]
        if (selected['phase']!=phase or selected['boundary_sha256']!=comparison['boundary_sha256']
            or selected['checkpoint_state_root']!=comparison['state_root'] or [e['path'] for e in selected['files']]!=FILES):
            raise EvidenceError('export phase boundary differs from fresh replay comparison')
        verify_inventory(directory/phase,selected['files'])
        root=replay_result[phase+'_model_root']
        if selected['model_root']!=root:raise EvidenceError('candidate export differs from continuously replayed weights')
        load_model(directory/phase,r,phase,expected_model_root=root)
        results[phase]={'model_root':root,'result':'PASS'}
    return {'schema':'ovl.replayed-export-check.v1','result':'PASS','registration_sha256':digest(r),
            'scope':'both exported model roots and inference configurations equal caller-supplied fresh full replay result',
            'models':results,'replay_report_sha256':digest(replay_result),'public_download_verification':'NOT_RUN',
            'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE','independent_third_party':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='action',required=True)
    for name in ('export','replay-check'):
        c=commands.add_parser(name)
        for arg in ('packet','registration-bundle','production-policy','source-policy','source-checkout','chain-directory',
                    'progress-directory','progress-policies','output'):
            c.add_argument('--'+arg,required=True,type=Path)
        if name=='export':c.add_argument('--prepared',required=True,type=Path)
        else:
            for arg in ('wikipedia-stream','conversation-stream','exports'):c.add_argument('--'+arg,required=True,type=Path)
    c=commands.add_parser('infer');c.add_argument('--directory',required=True,type=Path)
    c.add_argument('--registration',required=True,type=Path);c.add_argument('--registration-sha256',required=True)
    c.add_argument('--phase',choices=['base','chat'],required=True);c.add_argument('--prompt',required=True)
    c.add_argument('--max-new-tokens',type=int,default=64)
    a=p.parse_args()
    from .canonical import canonical
    try:
        if a.action=='infer':
            r=read_json(a.registration)
            if digest(r)!=a.registration_sha256:raise EvidenceError('registration differs from external selection')
            result=infer(a.directory,r,a.phase,a.prompt,max_new_tokens=a.max_new_tokens)
        else:
            from .anchoring import PublisherPolicy
            from .production_identity import ProductionPublisherPolicy
            from .progress_anchoring import ProgressPublisherPolicy
            args=(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
                  PublisherPolicy(**read_json(a.source_policy)),a.source_checkout,a.chain_directory,a.progress_directory,
                  [ProgressPublisherPolicy(**v) for v in read_json(a.progress_policies)])
            if a.action=='export':result=export_candidates(*args,a.prepared,a.output)
            else:
                from .production_replay import replay
                from .production_anchoring import object_at
                # No CLI accepts a saved PASS report as a substitute for replay.
                replay_result=replay(*args,{'wikipedia':a.wikipedia_stream,'conversation':a.conversation_stream},a.output)
                result=verify_replayed_exports(a.exports,object_at(a.packet,'registration.json'),replay_result)
                write_json(a.output/'exports-verification.json',result)
        print(canonical(result).decode());return 0
    except Exception as error:print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1


if __name__=='__main__':raise SystemExit(main())
