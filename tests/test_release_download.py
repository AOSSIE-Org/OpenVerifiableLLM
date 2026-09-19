"""Actual tiny safe models/greedy generation, fake anonymous host and identity."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import shutil
import pytest
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_release_commitment import setup as ancestry_setup
from ovl_pipeline import release_download as m,production_release,release_commitment,progress_commitment
from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,write_json

class Host:
    def __init__(self,value,payloads):
        self.value=value;self.files={};self.fetched=[];self.private=False
        for phase in ('base','chat'):
            self.files[value['models'][phase]['repo']]={n:(payloads[phase]/n).read_bytes() for n in production_release.PAYLOAD}
            self.files[value['models'][phase]['repo']].update({'release.json':canonical(value),'release.sigstore.json':b'{}','.gitattributes':b'*.safetensors filter=lfs\n'})
    def repo_info(self,repo,**kw):
        return SimpleNamespace(private=self.private,sha=kw['revision'],siblings=[SimpleNamespace(rfilename=n,size=len(b)) for n,b in self.files[repo].items()])
    def fetch(self,**kw):
        assert kw['token'] is False and kw['force_download'] is True and kw['local_files_only'] is False
        self.fetched.append((kw['repo_id'],kw['filename']))
        path=Path(kw['cache_dir'])/str(len(self.fetched));path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.files[kw['repo_id']][kw['filename']]);return path


def setup(inputs,prepared,tmp_path,monkeypatch,damage=None):
    r,value,run,downloads,reports=ancestry_setup(inputs,prepared,tmp_path,monkeypatch)
    payloads={ph:tmp_path/'payload'/ph for ph in ('base','chat')};host=Host(value,payloads)
    context=read_json(reports/'context.json');pp,sp,policies=release_commitment.policies(context)
    policy=production_release.ReleasePublisherPolicy(**{**asdict(pp),'workflow':production_release.RELEASE_WORKFLOW,'statement_sha256':digest(value)})
    def identity(statement,bundle,selected,**kw):
        if damage=='signature' or digest(read_json(statement))!=selected.statement_sha256:raise EvidenceError('explicit rejected identity double')
        return {'identity':'explicit-identity-test-double','statement_sha256':selected.statement_sha256}
    monkeypatch.setattr(m,'verify_anchor',identity);monkeypatch.setattr(production_release,'verify_anchor',identity)
    monkeypatch.setattr(m,'HfApi',lambda **kw:host);monkeypatch.setattr(m,'hf_hub_download',host.fetch)
    monkeypatch.setattr(progress_commitment,'download_archive',release_commitment.download_archive)
    selected={ph:{'repo':value['models'][ph]['repo'],'revision':'3'*40} for ph in ('base','chat')}
    if damage=='weight':host.files[selected['chat']['repo']]['model/model.safetensors']=b'altered'
    elif damage=='extra':host.files[selected['base']['repo']]['seed.key']=b'no'
    elif damage=='private':host.private=True
    elif damage=='tag':selected['base']['revision']='main'
    elif damage=='bundle-differs':host.files[selected['chat']['repo']]['release.sigstore.json']=b'{"different":true}'
    def verify(mode='artifacts',**kw):return m.verify(selected,policy,pp,sp,policies,tmp_path,tmp_path/'download',mode=mode,**kw)
    return r,value,host,selected,policy,verify


def test_fresh_public_models_full_parents_and_actual_greedy_inference_without_replay_claim(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    r,value,host,selected,policy,verify=setup(inputs,prepared,tmp_path,monkeypatch);result=verify()
    assert result['result']=='PASS' and result['locally_recomputed_training'] is False
    assert result['continuous_replay']=='NOT_RUN' and result['raw_reconstruction']=='NOT_RUN'
    assert result['complete_computation_sha256'] is None and result['independent_third_party'] is False
    assert len(host.fetched)==2*(len(production_release.PAYLOAD)+3)
    generated=read_json(tmp_path/'download/inference.json');assert generated['result']=='PASS'
    assert len(generated['models']['chat'])==2
    with pytest.raises(EvidenceError,match='fresh output'):verify()


@pytest.mark.parametrize('damage',['signature','weight','extra','private','tag','bundle-differs'])
def test_bad_public_releases_fail_closed_without_final_verification(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    r,value,host,selected,policy,verify=setup(inputs,prepared,tmp_path,monkeypatch,damage)
    with pytest.raises(EvidenceError):verify()
    assert not(tmp_path/'download/verification.json').exists()
    if damage=='signature':assert not any(n=='model/model.safetensors' for _,n in host.fetched)


@pytest.mark.parametrize('incompatible_inference',[False,True])
def test_full_mode_calls_actual_complete_transformations_and_continuous_cpu_fixture_replay(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,incompatible_inference):
    from ovl_pipeline import production_replay,production_export,runtime_launch
    r,value,host,selected,policy,verify=setup(inputs,prepared,tmp_path,monkeypatch);calls=[]
    def cpu_launch(lock,wheels,venv,code,out,module,args,**kw):
        options=dict(zip(args[1::2],args[2::2]));assert module=='ovl_pipeline.production_export' and args[0]=='replay-check'
        streams={p:Path(options['--'+p+'-stream']) for p in ('wikipedia','conversation')}
        assert all(p.is_relative_to(tmp_path/'download/computation/reconstructed') for p in streams.values())
        calls.append(streams)
        result=production_replay.replay(tmp_path/'download/parents/packet',None,None,None,tmp_path,
            Path(options['--chain-directory']),Path(options['--progress-directory']),[],streams,Path(options['--output']))
        production_export.verify_replayed_exports(Path(options['--exports']),r,result)
        return {'exit_code':0,'explicit-test-double':'CPU in-process launcher'}
    monkeypatch.setattr(runtime_launch,'launch',cpu_launch)
    if incompatible_inference:
        original=production_export.infer
        def incompatible(*a,**kw):
            result=original(*a,**kw);result['output_ids']=[999];result['runtime']={'explicit-test-double':'different CPU'};return result
        monkeypatch.setattr(production_export,'infer',incompatible)
    runtime={n:tmp_path/n for n in ('lock','wheels','venv','source','interpreter_archive','interpreter_root')}
    runtime.update(allowed_generated={},interpreter_sha256='1'*64)
    result=verify('full',raw=tmp_path/'raw',runtime=runtime)
    assert result['locally_recomputed_training'] is True and len(calls)==1
    assert result['result']==('UNSUPPORTED' if incompatible_inference else 'PASS')
    assert result['inference_result']==('UNSUPPORTED' if incompatible_inference else 'PASS')
    assert result['continuous_replay']=='PASS' and result['raw_reconstruction']=='PASS'
    reconstruction=read_json(tmp_path/'download/computation/reconstruction.json')
    assert reconstruction['stages_adopted_from_local_cache']==[] and len(reconstruction['stages_executed_this_run'])==6
    replay=read_json(tmp_path/'download/computation/numerical-replay/verification.json')
    assert replay['updates_recomputed']=={p:r['coverage'][p]['updates'] for p in ('wikipedia','conversation')}
    assert replay['prover_checkpoints_restored'] is False and result['independent_third_party'] is False


def test_changed_inference_is_not_accepted_as_compatible(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    from ovl_pipeline import production_export
    r,value,host,selected,policy,verify=setup(inputs,prepared,tmp_path,monkeypatch)
    actual=production_export.infer
    def changed(*a,**kw):
        result=actual(*a,**kw);result['output_ids']=[999];return result
    monkeypatch.setattr(production_export,'infer',changed)
    result=verify();assert result['result']=='FAIL' and result['inference_result']=='FAIL'
    assert result['continuous_replay']=='NOT_RUN' and result['locally_recomputed_training'] is False
