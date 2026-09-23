from pathlib import Path
import io
import tarfile
import pytest
from ovl_pipeline.canonical import EvidenceError, digest, file_hash, write_json
from ovl_pipeline.publication_privacy import review_export
from ovl_pipeline.publication_pause import require_publication_open


def fixture(tmp_path, name='record/report.json', data=b'{"result":"PASS"}', extra=None):
    stage=tmp_path/'staging'; stage.mkdir()
    member={'path':name,'bytes':len(data),'sha256':__import__('hashlib').sha256(data).hexdigest()}
    members=[member]
    with tarfile.open(stage/'retained-exports.tar.gz','w:gz') as tar:
        info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        if extra:
            info=tarfile.TarInfo(extra);info.size=2;tar.addfile(info,io.BytesIO(b'{}'))
    checkpoint={'schema':'ovl.sanitized-operational-summary.v1','scope':'Operator development computation only','result':'PASS'}
    write_json(stage/'checkpoint.json',checkpoint);write_json(stage/'retained-export-inventory.json',members)
    entries=[{'path':n,'bytes':(stage/n).stat().st_size,'sha256':file_hash(stage/n)} for n in ['checkpoint.json','retained-export-inventory.json','retained-exports.tar.gz']]
    plan={'kind':'operational-evidence','files':entries}
    review={'schema':'ovl.operational-publication-review.v1','plan_sha256':digest(plan),'files':entries,'archive_members':members,'checkpoint_fields':sorted(checkpoint)}
    path=tmp_path/'review.json';write_json(path,review)
    return plan,stage,path,review


def test_reviewed_technical_payload(tmp_path):
    plan,stage,path,_=fixture(tmp_path);review_export(plan,stage,path)


@pytest.mark.parametrize('name,data',[
    ('project/goal_state.json',b'{}'),('PROJECT_GOAL.md',b'goal'),
    ('advisory/spec.json',b'{}'),('preparation-recovery-advisory/initial-proposal.md',b'proposal'),('AGENTS.md',b'instructions'),('prompts/assignment.md',b'agent instructions'),
    ('record/report.json',b'{"account_balance_usd":"100"}'),
    ('record/report.json',b'{"balance_usd":"100"}'),('record/report.json',b'{"currentSpend":"100"}'),
    ('record/report.json',br'{"\u0061ccount_balance_usd":"100"}'),
    ('record/report.json',b'/home/alice/private/file'),
    ('record/state.safetensors',b'{"owner_instruction":"private"}'),
    ('record/input.tar.gz',b'opaque'),('../outside',b'{}')])
def test_review_cannot_admit_forbidden_material(tmp_path,name,data):
    plan,stage,path,_=fixture(tmp_path,name,data)
    with pytest.raises(EvidenceError):review_export(plan,stage,path)


def test_unreviewed_member_rejected(tmp_path):
    plan,stage,path,_=fixture(tmp_path,extra='record/extra.json')
    with pytest.raises(EvidenceError):review_export(plan,stage,path)


def test_allowlist_plan_mismatch(tmp_path):
    plan,stage,path,review=fixture(tmp_path);review['plan_sha256']='0'*64;write_json(path,review)
    with pytest.raises(EvidenceError):review_export(plan,stage,path)


def test_reviewed_bytes_changed(tmp_path):
    plan,stage,path,_=fixture(tmp_path);(stage/'retained-exports.tar.gz').write_bytes(b'changed')
    with pytest.raises(EvidenceError):review_export(plan,stage,path)


def test_public_conversation_data_is_not_operator_context(tmp_path):
    review_export({'kind':'raw'},tmp_path,tmp_path/'absent')
    review_export({'kind':'prepared'},tmp_path,tmp_path/'absent')


def test_pause_and_resume_without_modifying_real_pause(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    require_publication_open()
    marker=tmp_path/'.local/state/openverifiablellm/publication-paused.json';marker.parent.mkdir(parents=True);marker.write_text('{}')
    with pytest.raises(EvidenceError):require_publication_open()
    marker.unlink();require_publication_open()


def test_broken_pause_symlink_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    marker=tmp_path/'.local/state/openverifiablellm/publication-paused.json';marker.parent.mkdir(parents=True);marker.symlink_to(tmp_path/'missing')
    with pytest.raises(EvidenceError):require_publication_open()


def test_operational_publisher_requires_review_then_preserves_roundtrip(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import publish_evidence_archive as transport
    monkeypatch.setattr(Path,'home',lambda:tmp_path/'isolated-home')
    plan,stage,review_path,review=fixture(tmp_path)
    root=file_hash(stage/'checkpoint.json')
    plan.update(schema='ovl.evidence-publication-plan.v1',repo=transport.REPO,prefix='operational-evidence/'+root,subject_sha256=root)
    pp=tmp_path/'plan.json';write_json(pp,plan)
    class API:
        sha='a'*40
        files={}
        def repo_info(self,*args,**kwargs):return SimpleNamespace(private=False,sha=self.sha)
        def list_repo_files(self,*args,**kwargs):return sorted(self.files)
        def create_commit(self,*args,**kwargs):
            self.files={op.path_in_repo:op.path_or_fileobj.read() for op in kwargs['operations']}
            self.sha='b'*40
            return SimpleNamespace(oid=self.sha,commit_url='https://huggingface.co/placeholder')
        def fetch(self,**kwargs):
            f=Path(kwargs['cache_dir'])/kwargs['filename'];f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(self.files[kwargs['filename']]);return f
    api=API()
    with pytest.raises((EvidenceError,FileNotFoundError)):
        transport.upload(pp,stage,tmp_path/'missing-review',api=api)
    assert not api.files
    review['plan_sha256']=digest(plan);write_json(pp.with_name(pp.name+'.review.json'),review)
    # Keep the operational member review and the new exact semantic gate real;
    # replace only the external scanner/Git commands for this synthetic archive.
    import publication_export_gate as gate
    from test_publication_export_gate import approval,scanner
    approval(pp,stage,plan);original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(plan,[])))
    transport.upload(pp,stage,tmp_path/'upload',api=api)
    assert transport.download(pp,api.sha,tmp_path/'download',api=api,fetch_file=api.fetch)['result']=='PASS'


def add_example_review(path,review):
    entry=review['archive_members'][0]
    review['public_dependency_examples']=[dict(path=entry['path'],sha256=entry['sha256'],markers=['/home/example/'],source_url='https://raw.githubusercontent.com/example/library/'+'a'*40+'/source.py',source_sha256='b'*64,reason='Synthetic public documentation example for offline test')]
    write_json(path,review)


def test_exact_member_public_example_review(tmp_path):
    plan,stage,path,review=fixture(tmp_path,data=b'/home/example/documentation')
    add_example_review(path,review)
    review_export(plan,stage,path)


@pytest.mark.parametrize('change',['hash','path','source','account','extra-private','changed-bytes'])
def test_example_review_cannot_waive_other_content(tmp_path,change):
    data=b'/home/example/documentation'
    if change=='extra-private':data+=b' /home/another/private'
    if change=='account':data+=b' {"account_balance_usd":"100"}'
    plan,stage,path,review=fixture(tmp_path,data=data)
    add_example_review(path,review);item=review['public_dependency_examples'][0]
    if change=='hash':item['sha256']='0'*64
    if change=='path':item['path']='record/another.pyc'
    if change=='source':item['source_url']='https://example.invalid/unpinned'
    if change=='account':item['markers'].append('"account_balance_usd":')
    if change=='changed-bytes':
        # A newly reviewed plan cannot silently inherit an old member exception.
        review['archive_members'][0]['sha256']='c'*64
        write_json(stage/'retained-export-inventory.json',review['archive_members'])
        for entry in plan['files']:
            f=stage/entry['path'];entry.update(bytes=f.stat().st_size,sha256=file_hash(f))
        review['plan_sha256']=digest(plan)
    write_json(path,review)
    with pytest.raises(EvidenceError):review_export(plan,stage,path)
