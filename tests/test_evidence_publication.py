"""Closed immutable-prefix transport with a fake provider; no public mutation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publish_evidence_archive as pub
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json


@pytest.fixture(autouse=True)
def synthetic_privacy_gate(monkeypatch):
    # Transport unit tests use a named double. The actual fail-closed review and
    # installed scanner contract are exercised in test_publication_export_gate.
    import publication_export_gate
    monkeypatch.setattr(publication_export_gate,'require_review',lambda plan_path,*a,**kw:{'synthetic-test-double':True,'plan_sha256':digest(read_json(plan_path))})


def plan(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();statement={'synthetic':True};root=digest(statement)
    write_json(stage/'statement.json',statement);write_json(stage/'statement.sigstore.json',{'explicit-test-double':True})
    value={'schema':'ovl.evidence-publication-plan.v1','repo':pub.REPO,'kind':'progress-anchor',
           'prefix':'production-progress/'+'1'*64+'/progress-00000','subject_sha256':root,
           'files':inventory(stage,['statement.json','statement.sigstore.json'])}
    path=tmp_path/'plan.json';write_json(path,value);return path,stage,value

class Fake:
    def __init__(self):self.sha='2'*40;self.files={};self.commits=0;self.fail_after_commit=False;self.private=False
    def repo_info(self,*a,**kw):return SimpleNamespace(sha=kw.get('revision',self.sha),private=self.private)
    def list_repo_files(self,*a,**kw):return sorted(self.files)
    def create_commit(self,*a,**kw):
        assert kw['parent_commit']==self.sha
        self.commits+=1;self.files.update({e.path_in_repo:(e.path_or_fileobj.read() if hasattr(e.path_or_fileobj,'read') else Path(e.path_or_fileobj).read_bytes()) for e in kw['operations']});self.sha='3'*40
        if self.fail_after_commit:raise TimeoutError('unknown client completion')
        return SimpleNamespace(oid=self.sha,commit_url='https://huggingface.co/datasets/'+pub.REPO+'/commit/'+self.sha)
    def fetch(self,**kw):
        assert kw['token'] is False and kw['force_download'] is True and kw['local_files_only'] is False
        cache=Path(kw['cache_dir']);cache.mkdir(parents=True,exist_ok=True)
        file=cache/str(len(list(cache.iterdir())));file.write_bytes(self.files[kw['filename']]);return file


def test_write_ahead_upload_and_actual_complete_download(tmp_path):
    pp,stage,p=plan(tmp_path);api=Fake();result=pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert result['result']=='UPLOADED_NOT_DOWNLOAD_VERIFIED' and api.commits==1
    assert read_json(tmp_path/'upload/intent.json')['parent_revision']=='2'*40
    downloaded=pub.download(pp,api.sha,tmp_path/'download',api=api,fetch_file=api.fetch)
    assert downloaded['result']=='PASS' and downloaded['semantic_verification']=='NOT_RUN'
    assert inventory(tmp_path/'download/downloaded',sorted(x['path'] for x in p['files']))==p['files']
    assert (tmp_path/'download/downloaded/statement.json').stat().st_nlink==2
    with pytest.raises(EvidenceError):pub.upload(pp,stage,tmp_path/'another-upload',api=api)
    assert api.commits==1


def test_unknown_commit_result_reconciles_without_duplicate_publication(tmp_path):
    pp,stage,p=plan(tmp_path);api=Fake();api.fail_after_commit=True
    with pytest.raises(TimeoutError):pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert (tmp_path/'upload/intent.json').exists() and not(tmp_path/'upload/upload.json').exists()
    with pytest.raises(EvidenceError):pub.upload(pp,stage,tmp_path/'upload',api=api)
    result=pub.reconcile(pp,tmp_path/'upload',api=api)
    assert result['result']=='FOUND_AWAITING_COMPLETE_DOWNLOAD' and api.commits==1
    assert pub.download(pp,result['revision'],tmp_path/'download',api=api,fetch_file=api.fetch)['result']=='PASS'


@pytest.mark.parametrize('change',['bytes','missing','extra','private','tag'])
def test_actual_download_refuses_changed_or_unavailable_public_archive(tmp_path,change):
    pp,stage,p=plan(tmp_path);api=Fake();pub.upload(pp,stage,tmp_path/'upload',api=api);revision=api.sha
    if change=='bytes':api.files[p['prefix']+'/statement.json']=b'altered'
    elif change=='missing':del api.files[next(iter(api.files))]
    elif change=='extra':api.files[p['prefix']+'/unexpected']=b'extra'
    elif change=='private':api.private=True
    else:revision='main'
    with pytest.raises(EvidenceError):pub.download(pp,revision,tmp_path/'download',api=api,fetch_file=api.fetch)
    assert not(tmp_path/'download/verification.json').exists()


@pytest.mark.parametrize('change',['repo','path','root','duplicate','extra-staged','changed-staged'])
def test_publication_closed_plan_and_source_bytes_before_remote_mutation(tmp_path,change):
    pp,stage,p=plan(tmp_path);api=Fake()
    if change=='repo':p['repo']='other/repo'
    elif change=='path':p['prefix']='../../elsewhere'
    elif change=='root':p['subject_sha256']='0'*64
    elif change=='duplicate':p['files'].append(p['files'][0])
    elif change=='extra-staged':(stage/'secret').write_bytes(b'should-not-upload')
    else:(stage/'statement.json').write_bytes(b'changed')
    write_json(pp,p)
    with pytest.raises(EvidenceError):pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert api.commits==0


def test_reconciliation_absence_does_not_reissue_or_erase_intent(tmp_path):
    pp,stage,p=plan(tmp_path);api=Fake();pub.upload(pp,stage,tmp_path/'upload',api=api)
    before=(tmp_path/'upload/intent.json').read_bytes();api.files={}
    with pytest.raises(EvidenceError):pub.reconcile(pp,tmp_path/'upload',api=api)
    assert api.commits==1 and before==(tmp_path/'upload/intent.json').read_bytes()


def test_two_publishers_cannot_hold_same_plan_lease(tmp_path):
    import os
    pp,stage,p=plan(tmp_path);api=Fake();fd=pub.lease(pp.with_name(pp.name+'.publication.lock'))
    try:
        with pytest.raises(EvidenceError,match='lease'):pub.upload(pp,stage,tmp_path/'upload',api=api)
    finally:os.close(fd)
    assert api.commits==0


@pytest.mark.parametrize('kind',['prepared','checkpoint','registration-anchor'])
def test_other_supported_archive_kinds_preserve_closed_subject_roots(tmp_path,kind):
    stage=tmp_path/'stage';stage.mkdir();subject={'synthetic':kind};root=digest(subject)
    if kind=='prepared':
        write_json(stage/'preparation.json',subject);prefix='prepared/'+root
        folder=stage/'tokenizer';folder.mkdir();(folder/'tokenizer.json').write_bytes(b'{}')
        names=['preparation.json','tokenizer/tokenizer.json']
    elif kind=='checkpoint':
        write_json(stage/'checkpoint.json',subject);(stage/'state.json').write_bytes(b'{}');(stage/'state.safetensors').write_bytes(b'fake-only')
        prefix='production-checkpoints/'+'a'*64+'/boundary-00001';names=['checkpoint.json','state.json','state.safetensors']
    else:
        write_json(stage/'registration.sigstore.json',subject);prefix='production-anchors/'+root;names=['registration.sigstore.json']
    value={'schema':'ovl.evidence-publication-plan.v1','repo':pub.REPO,'kind':kind,'prefix':prefix,'subject_sha256':root,'files':inventory(stage,names)}
    pp=tmp_path/'plan.json';write_json(pp,value);api=Fake()
    assert pub.upload(pp,stage,tmp_path/'upload',api=api)['semantic_verification']=='NOT_RUN'
    assert pub.download(pp,api.sha,tmp_path/'download',api=api,fetch_file=api.fetch)['semantic_verification']=='NOT_RUN'


@pytest.mark.parametrize('kind',['registration-packet','release-evidence','release-anchor','python-runtime'])
def test_new_closed_evidence_kinds_have_complete_anonymous_byte_roundtrip(tmp_path,kind):
    from ovl_pipeline.production_anchoring import PACKET_FILES
    from ovl_pipeline.production_release import EVIDENCE
    stage=tmp_path/'stage';stage.mkdir()
    names={'registration-packet':sorted(PACKET_FILES),'release-evidence':EVIDENCE,
           'release-anchor':['release.json','release.sigstore.json'],
           'python-runtime':['SHA256SUMS','acquisition.json','distribution.tar.gz','payloads.json'],
           'operational-evidence':['checkpoint.json','retained-export-inventory.json','retained-exports.tar.gz']}[kind]
    for name in names:(stage/name).write_bytes(b'{}')
    files=inventory(stage,names)
    marker={'registration-packet':'registration.json','release-anchor':'release.json','python-runtime':'distribution.tar.gz',
            'operational-evidence':'checkpoint.json'}
    root=digest(files) if kind=='release-evidence' else next(e['sha256'] for e in files if e['path']==marker[kind])
    prefix={'registration-packet':'production-registration','release-evidence':'release-evidence',
            'release-anchor':'release-anchors','python-runtime':'runtime-python','operational-evidence':'operational-evidence'}[kind]+'/'+root
    value={'schema':'ovl.evidence-publication-plan.v1','repo':pub.REPO,'kind':kind,'prefix':prefix,'subject_sha256':root,'files':files}
    pp=tmp_path/'plan.json';write_json(pp,value);api=Fake()
    pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert pub.download(pp,api.sha,tmp_path/'download',api=api,fetch_file=api.fetch)['result']=='PASS'
    value['files'].pop();write_json(pp,value)
    with pytest.raises(EvidenceError):pub.validate(value)
