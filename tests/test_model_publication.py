"""Real safe payload checks, explicit publisher identity and writable-host doubles."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publish_models as m
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_release_download import setup as download_setup
from ovl_pipeline.canonical import EvidenceError,read_json,write_json

class Host:
    def __init__(self):
        self.repositories={};self.creates=0;self.commits=0;self.lose_create=False;self.lose_commit=False;self.fail_read_after_create=False
    def repo_exists(self,repo,**kw):return repo in self.repositories
    def create_repo(self,repo,**kw):
        assert kw=={'repo_type':'model','private':False,'exist_ok':False}
        assert repo not in self.repositories
        self.creates+=1;self.repositories[repo]={'sha':'1'*40,'files':{'.gitattributes':b'attributes'}}
        if self.lose_create:raise TimeoutError('lost creation response')
    def repo_info(self,repo,**kw):
        if self.fail_read_after_create:raise TimeoutError('failed read after create')
        return SimpleNamespace(sha=self.repositories[repo]['sha'],private=False)
    def list_repo_files(self,repo,**kw):return sorted(self.repositories[repo]['files'])
    def create_commit(self,repo,**kw):
        state=self.repositories[repo];assert kw['parent_commit']==state['sha'];self.commits+=1
        state['files'].update({op.path_in_repo:Path(op.path_or_fileobj).read_bytes() for op in kw['operations']});state['sha']='2'*40
        if self.lose_commit:raise TimeoutError('lost commit response')
        return SimpleNamespace(oid=state['sha'],commit_url='https://huggingface.co/'+repo+'/commit/'+state['sha'])


def setup(inputs,prepared,tmp_path,monkeypatch,damage=None):
    r,value,remote,selected,policy,verify=download_setup(inputs,prepared,tmp_path,monkeypatch,damage)
    statement=tmp_path/'release.json';write_json(statement,value);bundle=tmp_path/'release.sigstore.json';write_json(bundle,{})
    payloads={p:tmp_path/'payload'/p for p in ('base','chat')};host=Host()
    def run(phase='base',out=None):return m.publish(phase,statement,bundle,policy,payloads,out or tmp_path/'publication',api=host)
    return value,host,run


def test_new_repository_only_and_preserved_write_ahead_without_download_credit(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch);result=run()
    assert result['result']=='UPLOADED_NOT_DOWNLOAD_VERIFIED' and host.creates==host.commits==1
    assert result['training_recomputed_by_publication'] is False
    assert set(host.repositories[result['repo']]['files'])==set(m.NAMES+['.gitattributes'])
    with pytest.raises(EvidenceError):run()
    with pytest.raises(EvidenceError,match='already exists'):run(out=tmp_path/'second-attempt')
    assert host.creates==host.commits==1
    chat=run('chat',tmp_path/'chat-publication');assert chat['repo']!=result['repo'] and host.creates==host.commits==2


def test_uncertain_commit_recovers_without_another_mutation(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch);host.lose_commit=True
    with pytest.raises(TimeoutError):run()
    out=tmp_path/'publication';original=(out/'commit-intent.json').read_bytes()
    with pytest.raises(EvidenceError):m.resume_unattempted_commit(out,api=host)
    recovered=m.reconcile(out,api=host)
    assert recovered['result']=='FOUND_AWAITING_COMPLETE_DOWNLOAD' and recovered['provider_mutation']=='NOT_RUN'
    assert host.creates==host.commits==1 and (out/'commit-intent.json').read_bytes()==original


def test_uncertain_creation_preserves_empty_repo_and_does_not_guess_ownership(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch);host.lose_create=True
    with pytest.raises(TimeoutError):run()
    assert (tmp_path/'publication/intent.json').exists() and not(tmp_path/'publication/created.json').exists()
    with pytest.raises((EvidenceError,OSError)):m.resume_unattempted_commit(tmp_path/'publication',api=host)
    with pytest.raises(EvidenceError):run(out=tmp_path/'new-output')
    assert host.creates==1 and host.commits==0


def test_successful_creation_can_resume_only_if_commit_was_never_attempted(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch);actual=m.commit_pending
    monkeypatch.setattr(m,'commit_pending',lambda *a:(_ for _ in ()).throw(InterruptedError()))
    with pytest.raises(InterruptedError):run()
    assert (tmp_path/'publication/created.json').exists() and not(tmp_path/'publication/commit-intent.json').exists()
    monkeypatch.setattr(m,'commit_pending',actual)
    result=m.resume_unattempted_commit(tmp_path/'publication',api=host)
    assert result['result']=='UPLOADED_NOT_DOWNLOAD_VERIFIED' and host.creates==host.commits==1


@pytest.mark.parametrize('damage',['signature','extra','bytes','existing-destination'])
def test_no_remote_write_for_untrusted_or_unregistered_payload(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch,'signature' if damage=='signature' else None)
    if damage=='extra':(tmp_path/'payload/base/seed.key').write_bytes(b'excluded synthetic secret')
    elif damage=='bytes':(tmp_path/'payload/chat/model/model.safetensors').write_bytes(b'wrong')
    elif damage=='existing-destination':host.repositories[value['models']['base']['repo']]={'sha':'9'*40,'files':{'README.md':b'unrelated'}}
    with pytest.raises(EvidenceError):run()
    assert host.creates==host.commits==0
    if damage=='existing-destination':assert host.repositories[value['models']['base']['repo']]['files']=={'README.md':b'unrelated'}


@pytest.mark.parametrize('occupied',[False,True])
def test_destination_preflight_is_read_only_and_records_occupation(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,occupied):
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch)
    if occupied:host.repositories[value['models']['chat']['repo']]={'sha':'9'*40,'files':{'README.md':b'preserved'}}
    output=tmp_path/'preflight.json'
    if occupied:
        with pytest.raises(EvidenceError):m.check_destinations(tmp_path/'release.json',output,api=host)
    else:assert m.check_destinations(tmp_path/'release.json',output,api=host)['result']=='AVAILABLE'
    assert read_json(output)['result']==('UNAVAILABLE' if occupied else 'AVAILABLE')
    assert host.creates==host.commits==0


def test_uncertain_presend_commit_preserves_old_repo_and_allows_new_publication_identity(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    from ovl_pipeline import production_release as release
    value,host,run=setup(inputs,prepared,tmp_path,monkeypatch)
    def presend(*a,**kw):raise TimeoutError('commit never reached provider')
    monkeypatch.setattr(host,'create_commit',presend)
    with pytest.raises(TimeoutError):run()
    with pytest.raises(EvidenceError):m.reconcile(tmp_path/'publication',api=host)
    assert host.repositories[value['models']['base']['repo']]['files']=={'.gitattributes':b'attributes'}
    # Same training registration may select a fresh release-2 before new signing.
    assert release.model_repo(read_json(tmp_path/'payload/base/registration.json'),'base','release-2') not in host.repositories
    assert host.creates==1 and host.commits==0
