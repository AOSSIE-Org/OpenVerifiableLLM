"""Real tiny CPU model/state/data with explicit publisher/CUDA substitutes."""
from copy import deepcopy
import pytest
import torch
from safetensors.torch import load_file,save_file
from test_preparation import inputs
from test_prepared_verification import prepared
from test_production_chain import actual_artifacts,resign
from test_gpu_pilot import cpu_runtime
from ovl_pipeline import production_export as export,production_replay
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json,file_hash,inventory
from ovl_pipeline.training import code_root


def setup(prepared,tmp_path,monkeypatch):
    directory,value,source=prepared
    r,_,envs,key,checkpoints,streams=actual_artifacts((directory,value),tmp_path)
    r.update(preparation_sha256=digest(value),source_statement_sha256=source,code_root=code_root())
    r['runtime']['compatible_environment_sha256']=digest({'test_runtime':'CPU-substitute'})
    for e in envs:e['body']['registration']=digest(r)
    envs[0]['body']['previous']=digest(r);resign(envs,key)
    def authenticate(*a):return r,envs,{'explicit-publisher-test-double':True}
    monkeypatch.setattr(export,'authenticate',authenticate);monkeypatch.setattr(production_replay,'authenticate',authenticate)
    output=tmp_path/'export'
    def run():return export.export_candidates(None,None,None,None,None,checkpoints,None,None,directory,output)
    return r,envs,checkpoints,streams,output,run


def test_both_exports_equal_actual_fresh_continuous_replay_and_generate(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,envs,checkpoints,streams,out,run=setup(prepared,tmp_path,monkeypatch)
    result=run();assert result['status']=='EXPORTED_NOT_TRAINING_VERIFIED'
    replay=production_replay.replay(None,None,None,None,None,checkpoints,None,None,streams,tmp_path/'replay')
    checked=export.verify_replayed_exports(out,r,replay);assert checked['result']=='PASS'
    for phase in ('base','chat'):
        a=export.infer(out/phase,r,phase,'A factual claim?',max_new_tokens=4,expected_model_root=replay[phase+'_model_root'])
        b=export.infer(out/phase,r,phase,'A factual claim?',max_new_tokens=4,expected_model_root=replay[phase+'_model_root'])
        assert a==b and a['factual_accuracy']=='NOT_ESTABLISHED_BY_PROVENANCE'
    with pytest.raises(EvidenceError,match='fresh'):run()


def test_gpu_rng_is_never_restored_by_cpu_weight_export(prepared,tmp_path,monkeypatch):
    r,envs,checkpoints,streams,out,run=setup(prepared,tmp_path,monkeypatch)
    import ovl_pipeline.state as state
    monkeypatch.setattr(state,'restore',lambda *a:pytest.fail('export must not restore optimizer or device RNG'))
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda *a:pytest.fail('export cannot initialize CUDA RNG'))
    assert run()['status']=='EXPORTED_NOT_TRAINING_VERIFIED'


@pytest.mark.parametrize('mutation',['weights','template','controls','tokenizer','preparation','extra-pickle','symlink','wrong-phase','wrong-loader'])
def test_modified_model_or_inference_inputs_fail_closed(prepared,tmp_path,monkeypatch,mutation):
    r,envs,checkpoints,streams,out,run=setup(prepared,tmp_path,monkeypatch);run();path=out/'chat';config=read_json(path/'config.json')
    if mutation=='weights':
        ts=load_file(str(path/'model.safetensors'));name=next(iter(ts));ts[name]=ts[name]+1;save_file(ts,str(path/'model.safetensors'))
    elif mutation=='template':config['inference']['input_format']='bos-text-v1'
    elif mutation=='controls':config['tokenizer']['controls']['assistant']=55
    elif mutation=='tokenizer':
        # Even rehashing the self-described tokenizer config/manifest cannot
        # change the externally registered preparation root.
        data=read_json(path/'tokenizer.json');data['truncation']={'max_length':2,'stride':0,'strategy':'LongestFirst','direction':'Right'}
        write_json(path/'tokenizer.json',data);tm=read_json(path/'tokenizer-manifest.json');tm['files']=inventory(path,['tokenizer.json']);write_json(path/'tokenizer-manifest.json',tm)
        config['tokenizer']['sha256']=file_hash(path/'tokenizer.json');config['tokenizer']['manifest_sha256']=file_hash(path/'tokenizer-manifest.json')
    elif mutation=='preparation':write_json(path/'preparation.json',{'forged':'parent'})
    elif mutation=='extra-pickle':(path/'unsafe.pkl').write_bytes(b'not loaded')
    elif mutation=='symlink':(path/'model.safetensors').rename(tmp_path/'weights');(path/'model.safetensors').symlink_to(tmp_path/'weights')
    elif mutation=='wrong-phase':config['phase']='base'
    else:r['code_root']='0'*64
    write_json(path/'config.json',config)
    with pytest.raises(EvidenceError):export.load_model(path,r,'chat')


def test_matching_export_self_hash_cannot_replace_replayed_weights(cpu_runtime,prepared,tmp_path,monkeypatch):
    r,envs,checkpoints,streams,out,run=setup(prepared,tmp_path,monkeypatch);run()
    replay=production_replay.replay(None,None,None,None,None,checkpoints,None,None,streams,tmp_path/'replay')
    # A trusted replay root remains decisive even if local export records lie.
    report=read_json(out/'export.json');report['exports']['chat']['model_root']='a'*64;write_json(out/'export.json',report)
    with pytest.raises(EvidenceError,match='continuously replayed'):export.verify_replayed_exports(out,r,replay)


def test_invalid_anchor_never_reads_checkpoint_or_allocates_model(tmp_path,monkeypatch):
    def refused(*a):raise EvidenceError('missing public commitment')
    monkeypatch.setattr(export,'authenticate',refused)
    monkeypatch.setattr(export,'model_for_inference',lambda *a:pytest.fail('must authenticate first'))
    with pytest.raises(EvidenceError,match='public commitment'):
        export.export_candidates(None,None,None,None,None,None,None,None,None,tmp_path/'out')
    assert not(tmp_path/'out').exists()


@pytest.mark.parametrize('change',['zero-updates','missing-comparison','changed-boundary'])
def test_incomplete_replay_or_wrong_export_boundary_fails(cpu_runtime,prepared,tmp_path,monkeypatch,change):
    r,envs,checkpoints,streams,out,run=setup(prepared,tmp_path,monkeypatch);run()
    replay=production_replay.replay(None,None,None,None,None,checkpoints,None,None,streams,tmp_path/'replay')
    if change=='zero-updates':replay['updates_recomputed']['wikipedia']=0
    elif change=='missing-comparison':replay['comparisons'].pop()
    else:
        report=read_json(out/'export.json');report['exports']['base']['boundary_sha256']='a'*64;write_json(out/'export.json',report)
    with pytest.raises(EvidenceError):export.verify_replayed_exports(out,r,replay)


def test_cli_inference_uses_caller_selected_registration(prepared,tmp_path,monkeypatch,capsys):
    import sys
    r,envs,checkpoints,streams,out,run=setup(prepared,tmp_path,monkeypatch);run();rp=tmp_path/'registration.json';write_json(rp,r)
    args=['production_export','infer','--directory',str(out/'chat'),'--registration',str(rp),'--registration-sha256',digest(r),
          '--phase','chat','--prompt','Hello','--max-new-tokens','2']
    monkeypatch.setattr(sys,'argv',args);assert export.main()==0
    assert 'NOT_ESTABLISHED_BY_PROVENANCE' in capsys.readouterr().out
    args[7]='a'*64
    assert export.main()==1 and 'external selection' in capsys.readouterr().out
