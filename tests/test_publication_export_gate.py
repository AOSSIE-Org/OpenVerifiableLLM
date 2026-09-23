"""Synthetic exact-review/export-scan boundaries; no public provider calls."""
from pathlib import Path
from types import SimpleNamespace
import json,os,sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publication_export_gate as gate
import publish_evidence_archive as pub
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json


def fixture(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();write_json(stage/'statement.json',{'synthetic':True})
    write_json(stage/'statement.sigstore.json',{'test-double':True})
    plan={'schema':'ovl.evidence-publication-plan.v1','repo':pub.REPO,'kind':'progress-anchor',
          'prefix':'production-progress/'+'a'*64+'/progress-00000',
          'subject_sha256':digest({'synthetic':True}),'files':inventory(stage,['statement.json','statement.sigstore.json'])}
    pp=tmp_path/'plan.json';write_json(pp,plan)
    return pp,stage,plan


def approval(pp,stage,plan):
    request={'schema':'ovl.local-export-review-request.v1','plan_sha256':digest(plan),
             'destination':'https://huggingface.co/datasets/'+plan['repo'],'prefix':plan['prefix'],
             'staging':str(stage),'files':plan['files']}
    rp=pp.with_name(pp.name+'.privacy-review.json')
    write_json(rp,{'schema':'ovl.local-export-review.v1','request_sha256':digest(request),'result':'PASS',
                   'repository_context':str(pp.parent),'content_review':'Reviewed synthetic two-file fixture.'})
    rp.chmod(0o600);return rp


def scanner(plan,seen,*,wrong=False,alter=None):
    def run(args,**kw):
        seen.append(args);assert 0<kw['timeout']<=600
        if args[0]=='git':return SimpleNamespace(stdout='https://huggingface.co/datasets/'+plan['repo']+'\n')
        assert args[0:2]==['publication-privacy','export']
        if alter:alter()
        return SimpleNamespace(stdout=json.dumps({'schema':'local.privacy-scan.v1','result':'PASS',
            'files':[] if wrong else plan['files']}))
    return run


def test_exact_review_installed_scan_and_private_receipt(tmp_path):
    pp,stage,p=fixture(tmp_path);rp=approval(pp,stage,p);seen=[]
    result=gate.require_review(pp,stage,execute=scanner(p,seen))
    assert result['result']=='PASS' and len(seen)==2
    assert inventory(stage,[e['path'] for e in p['files']])==p['files']
    assert pp.with_name(pp.name+'.privacy-scan.json').exists()


@pytest.mark.parametrize('damage',['absent','content','plan','review-hash','mode','symlink','hardlink','extra','wrong-scan','changed-during-scan','wrong-destination'])
def test_review_cannot_admit_changed_or_unrelated_bytes(tmp_path,damage):
    pp,stage,p=fixture(tmp_path);rp=approval(pp,stage,p);seen=[]
    if damage=='absent':rp.unlink()
    elif damage=='content':(stage/'statement.json').write_text('different')
    elif damage=='plan':p['prefix']='production-progress/'+'b'*64+'/progress-00000';write_json(pp,p)
    elif damage=='review-hash':v=read_json(rp);v['request_sha256']='0'*64;write_json(rp,v)
    elif damage=='mode':rp.chmod(0o644)
    elif damage=='symlink':original=rp.with_suffix('.saved');rp.rename(original);rp.symlink_to(original)
    elif damage=='hardlink':os.link(stage/'statement.json',tmp_path/'shared')
    elif damage=='extra':(stage/'unreviewed').write_text('extra')
    runner=scanner(p,seen,wrong=damage=='wrong-scan',alter=(lambda:(stage/'statement.json').write_text('changed')) if damage=='changed-during-scan' else None)
    if damage=='wrong-destination':runner=lambda *a,**kw:SimpleNamespace(stdout='https://huggingface.co/datasets/other/example')
    with pytest.raises(EvidenceError):gate.require_review(pp,stage,execute=runner)
    assert not pp.with_name(pp.name+'.privacy-scan.json').exists()


def test_wait_is_bounded_and_valid_review_can_arrive(tmp_path):
    pp,stage,p=fixture(tmp_path);clock=[100];seen=[]
    def sleep(n):clock[0]+=n;approval(pp,stage,p)
    assert gate.require_review(pp,stage,deadline=110,wall=lambda:clock[0],monotonic=lambda:clock[0],sleep=sleep,execute=scanner(p,seen))['result']=='PASS'
    assert clock[0]==102


def test_expired_wait_and_wall_rollback_do_not_extend_deadline(tmp_path):
    pp,stage,p=fixture(tmp_path);clock=[100];seen=[]
    def sleep(n):clock[0]+=n
    with pytest.raises(EvidenceError,match='deadline'):
        gate.require_review(pp,stage,deadline=103,wall=lambda:100,monotonic=lambda:clock[0],sleep=sleep,execute=scanner(p,seen))
    assert clock[0]==103 and not seen


def test_scanner_timeout_does_not_produce_pass(tmp_path):
    import subprocess
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p)
    def run(args,**kw):
        if args[0]=='git':return SimpleNamespace(stdout='https://huggingface.co/datasets/'+p['repo'])
        raise subprocess.TimeoutExpired(args,kw['timeout'])
    with pytest.raises(EvidenceError,match='scan failed'):gate.require_review(pp,stage,execute=run)
    assert not pp.with_name(pp.name+'.privacy-scan.json').exists()


def test_upload_reaches_no_provider_before_review(tmp_path):
    pp,stage,p=fixture(tmp_path)
    class Forbidden:
        def repo_info(self,*args,**kwargs):raise AssertionError('provider reached before review')
    with pytest.raises(EvidenceError,match='semantic publication review'):
        pub.upload(pp,stage,tmp_path/'upload',api=Forbidden())
    assert not(tmp_path/'upload').exists()


def test_completed_review_progress_precedes_provider_without_upload_credit(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);seen=[];events=[];original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,seen)))
    class Ordered(Fake):
        def repo_info(self,*args,**kwargs):
            assert len(events)==1 and len(seen)==2
            return super().repo_info(*args,**kwargs)
    api=Ordered()
    def reviewed(identity):
        assert api.commits==0 and not(tmp_path/'upload').exists()
        receipt=read_json(pp.with_name(pp.name+'.privacy-scan.json'))
        assert identity=={'plan_sha256':digest(p),'privacy_gate_sha256':digest(receipt)}
        events.append(identity)
    pub.upload(pp,stage,tmp_path/'upload',api=api,reviewed=reviewed)
    assert len(events)==1 and api.commits==1


@pytest.mark.parametrize('failure',['no-review','wrong-inventory','expired','binding-stop'])
def test_wait_failure_or_stop_cannot_advance_publication(tmp_path,monkeypatch,failure):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);seen=[];events=[];original=gate.require_review
    if failure!='no-review':approval(pp,stage,p)
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,
        execute=scanner(p,seen,wrong=failure=='wrong-inventory'),wall=lambda:100,monotonic=lambda:100))
    api=Fake()
    def reviewed(identity):
        if failure=='binding-stop':raise EvidenceError('original controller requests stop')
        events.append(identity)
    with pytest.raises(EvidenceError):
        pub.upload(pp,stage,tmp_path/'upload',api=api,deadline=99 if failure=='expired' else None,reviewed=reviewed)
    assert events==[] and api.commits==0 and not(tmp_path/'upload').exists()


@pytest.mark.parametrize('damage',['mode','hardlink'])
def test_review_metadata_changed_during_scan_blocks_progress_and_upload(tmp_path,monkeypatch,damage):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);rp=approval(pp,stage,p);seen=[];events=[];original=gate.require_review
    def change():
        if damage=='mode':rp.chmod(0o644)
        else:os.link(rp,tmp_path/'linked-review')
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,seen,alter=change)))
    api=Fake()
    with pytest.raises(EvidenceError,match='owner-private'):
        pub.upload(pp,stage,tmp_path/'upload',api=api,reviewed=events.append)
    assert not events and api.commits==0 and not pp.with_name(pp.name+'.privacy-scan.json').exists()


@pytest.mark.parametrize('when',['provider-read','prefix-read','payload-copy'])
def test_stop_after_progress_before_remote_mutation_is_binding(tmp_path,monkeypatch,when):
    from contextlib import contextmanager
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);seen=[];events=[];stopped=[];original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,seen)))
    class Provider(Fake):
        def repo_info(self,*a,**kw):
            if when=='provider-read':stopped.append(True)
            return super().repo_info(*a,**kw)
        def list_repo_files(self,*a,**kw):
            if when=='prefix-read':stopped.append(True)
            return super().list_repo_files(*a,**kw)
    original_copy=gate.frozen_payloads
    @contextmanager
    def copy(*a,**kw):
        with original_copy(*a,**kw) as payloads:
            if when=='payload-copy':stopped.append(True)
            yield payloads
    monkeypatch.setattr(gate,'frozen_payloads',copy)
    def guard():
        if stopped:raise EvidenceError('binding stop')
    api=Provider()
    with pytest.raises(EvidenceError,match='binding stop'):
        pub.upload(pp,stage,tmp_path/'upload',api=api,guard=guard,reviewed=events.append)
    assert len(events)==1 and api.commits==0 and not (tmp_path/'upload/upload.json').exists()


def test_scan_finishing_after_deadline_is_not_accepted(tmp_path):
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);clock=[100];seen=[]
    run=scanner(p,seen,alter=lambda:clock.__setitem__(0,112))
    with pytest.raises(EvidenceError,match='deadline'):
        gate.require_review(pp,stage,deadline=110,wall=lambda:100,monotonic=lambda:clock[0],execute=run)
    assert not pp.with_name(pp.name+'.privacy-scan.json').exists()


def test_upload_rechecks_original_clock_after_provider_read(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);clock=[100];seen=[]
    original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,seen),wall=lambda:100,monotonic=lambda:clock[0]))
    monkeypatch.setattr(pub.time,'time',lambda:100)
    monkeypatch.setattr(pub.time,'monotonic',lambda:clock[0])
    class Slow(Fake):
        def repo_info(self,*a,**kw):clock[0]=112;return super().repo_info(*a,**kw)
    api=Slow()
    with pytest.raises(EvidenceError,match='deadline'):pub.upload(pp,stage,tmp_path/'upload',api=api,deadline=110)
    assert api.commits==0 and not(tmp_path/'upload').exists()


def test_upload_keeps_gate_receipt_out_of_public_files(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);seen=[];original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,seen)))
    api=Fake();pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert sorted(api.files)==sorted(p['prefix']+'/'+e['path'] for e in p['files'])
    assert read_json(tmp_path/'upload/intent.json')['privacy_gate_sha256']


def test_gate_cannot_approve_a_different_plan_than_upload(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);original=gate.require_review
    def switched(*a,**kw):
        other={**p,'prefix':'production-progress/'+'b'*64+'/progress-00000'}
        write_json(pp,other);approval(pp,stage,other)
        return original(*a,**kw,execute=scanner(other,[]))
    monkeypatch.setattr(gate,'require_review',switched);api=Fake()
    with pytest.raises(EvidenceError,match='different upload plan'):pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert api.commits==0


@pytest.mark.parametrize('replace_inode',[False,True])
def test_changed_source_after_scan_cannot_be_transmitted(tmp_path,monkeypatch,replace_inode):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,[])))
    class Mutates(Fake):
        def repo_info(self,*a,**kw):
            target=stage/'statement.sigstore.json';data=b'x'*target.stat().st_size
            if replace_inode:target.unlink()
            target.write_bytes(data);return super().repo_info(*a,**kw)
    api=Mutates()
    with pytest.raises(EvidenceError,match='differs from reviewed bytes'):pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert api.commits==0


def test_sdk_consumes_readonly_unlinked_snapshot_when_source_changes(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,[])))
    expected={p['prefix']+'/'+e['path']:(stage/e['path']).read_bytes() for e in p['files']}
    class Mutates(Fake):
        def create_commit(self,*a,**kw):
            for op in kw['operations']:
                assert os.fstat(op.path_or_fileobj.fileno()).st_nlink==0
                with pytest.raises(OSError):os.write(op.path_or_fileobj.fileno(),b'bad')
            (stage/'statement.sigstore.json').write_bytes(b'changed during SDK read')
            return super().create_commit(*a,**kw)
    api=Mutates();pub.upload(pp,stage,tmp_path/'upload',api=api)
    assert api.files==expected


def test_operation_construction_cannot_cross_deadline_then_commit(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);original=gate.require_review;clock=[100]
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,[]),wall=lambda:100,monotonic=lambda:clock[0]))
    monkeypatch.setattr(pub.time,'time',lambda:100);monkeypatch.setattr(pub.time,'monotonic',lambda:clock[0])
    constructor=pub.CommitOperationAdd
    def slow(**kw):clock[0]=112;return constructor(**kw)
    monkeypatch.setattr(pub,'CommitOperationAdd',slow);api=Fake()
    with pytest.raises(EvidenceError,match='deadline'):pub.upload(pp,stage,tmp_path/'upload',api=api,deadline=110)
    assert api.commits==0


def test_intent_uses_selected_plan_even_if_path_changes_at_save(tmp_path,monkeypatch):
    from test_evidence_publication import Fake
    pp,stage,p=fixture(tmp_path);approval(pp,stage,p);original=gate.require_review
    monkeypatch.setattr(gate,'require_review',lambda *a,**kw:original(*a,**kw,execute=scanner(p,[])))
    write=pub.write_json
    def replace(path,value):
        if path.name=='intent.json':write(pp,{**p,'prefix':'production-progress/'+'b'*64+'/progress-00000'})
        write(path,value)
    monkeypatch.setattr(pub,'write_json',replace);api=Fake();pub.upload(pp,stage,tmp_path/'upload',api=api)
    intent=read_json(tmp_path/'upload/intent.json');assert intent['plan_sha256']==digest(intent['plan'])==digest(p)
    write(pp,p)
    assert pub.reconcile(pp,tmp_path/'upload',api=api)['result']=='FOUND_AWAITING_COMPLETE_DOWNLOAD'
    assert api.commits==1


def test_snapshots_close_on_sdk_failure_and_keep_sources(tmp_path):
    pp,stage,p=fixture(tmp_path);handles=[];before={e['path']:(stage/e['path']).read_bytes() for e in p['files']}
    with pytest.raises(RuntimeError):
        with gate.frozen_payloads(stage,p['files'],tmp_path,lambda:None) as files:
            handles.extend(handle for _,handle in files);raise RuntimeError('synthetic SDK failure')
    assert handles and all(h.closed for h in handles)
    assert {e['path']:(stage/e['path']).read_bytes() for e in p['files']}==before
    assert sorted(x.name for x in tmp_path.iterdir())==['plan.json','stage']


def test_real_sdk_repeated_and_threaded_stream_reads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hashlib import sha256
    pp,stage,p=fixture(tmp_path)
    with gate.frozen_payloads(stage,p['files'],tmp_path,lambda:None) as files:
        operations=[pub.CommitOperationAdd(path_in_repo=e['path'],path_or_fileobj=f) for e,f in files]
        def consume(op):
            results=[]
            for _ in range(2):
                with op.as_file() as stream:results.append(sha256(stream.read()).hexdigest())
            return results
        with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(consume,operations))
        assert results==[[e['sha256']]*2 for e in p['files']]
