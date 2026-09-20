"""Registration handoff with real public-byte transport doubles, never signer credit."""
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publish_production_registration as m
import publish_progress_boundary as shared
from test_production_commitment import request
from test_evidence_publication import Fake
from ovl_pipeline import production_anchoring,production_commitment
from ovl_pipeline.anchoring import PublisherPolicy
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,write_json


def configured(tmp_path,monkeypatch):
    packet,req=request(tmp_path);r=read_json(packet/'registration.json');source=PublisherPolicy(**req['source_policy'])
    calls=[];provider=Fake()
    upload=shared.transport.upload;download=shared.transport.download;reconcile=shared.transport.reconcile
    monkeypatch.setattr(shared.transport,'upload',lambda *a:upload(*a,api=provider))
    monkeypatch.setattr(shared.transport,'download',lambda *a:download(*a,api=provider,fetch_file=provider.fetch))
    monkeypatch.setattr(shared.transport,'reconcile',lambda *a:reconcile(*a,api=provider))
    monkeypatch.setattr(m,'verify_code',lambda *a:{'explicit-source-test-double':True})
    monkeypatch.setattr(production_commitment,'verify_code',lambda *a:{'explicit-source-test-double':True})
    def signature(statement,bundle,policy,**kw):
        calls.append(('signature',policy.workflow,policy.statement_sha256))
        assert file_hash(statement)==policy.statement_sha256
        if policy.workflow==m.PRODUCTION_WORKFLOW:
            selected=read_json(bundle)
            if selected!={'explicit-test-bundle-policy':asdict(policy)}:raise EvidenceError('explicit signature test double rejected')
        return {'scope':'explicit-identity-test-double-not-real-signature'}
    monkeypatch.setattr(production_anchoring,'verify_anchor',signature)
    def commit(req,reg,out,**kw):
        assert digest(reg)==req['registration_sha256'];calls.append(('request',digest(req)));return '2'*40
    monkeypatch.setattr(m,'request_commit',commit)
    def action(revision,out,deadline,**kw):
        out.mkdir();write_json(out/'registration.sigstore.json',{'explicit-test-bundle-policy':asdict(m.expected_policy(revision,r))})
        return {'run_id':1,'revision':revision,'attempt':1,'artifact_name':f'pipeline-production-registration-{revision}-1'}
    monkeypatch.setattr(m,'actions_artifact',action)
    return packet,r,source,provider,calls


def test_publication_and_restart_reverify_actual_bytes_and_external_policy(tmp_path,monkeypatch):
    packet,r,source,provider,calls=configured(tmp_path,monkeypatch);out=tmp_path/'publisher';deadline=int(time.time())+300
    first=m.publish(packet,digest(r),source,tmp_path,out,deadline)
    assert first['result']=='PASS' and first['production_admission']=='NOT_RUN' and provider.commits==2
    config=read_json(out/'progress-publisher-config.json')
    assert config['registration_request']['packet']==first['packet']
    assert config['registration_anchor']==first['registration_anchor']
    previous=len(calls);assert m.publish(packet,digest(r),source,tmp_path,out,deadline)==first
    assert provider.commits==2 and len(calls)>previous and len(list(out.glob('verification-*.json')))==2
    assert first['production_policy']['source_revision']=='2'*40
    assert first['production_policy']['workflow']==m.PRODUCTION_WORKFLOW


def test_registration_reports_completed_finite_gates_only(tmp_path,monkeypatch):
    packet,r,source,provider,calls=configured(tmp_path,monkeypatch);events=[]
    m.publish(packet,digest(r),source,tmp_path,tmp_path/'publisher',int(time.time())+300,
              progress=lambda stage,identity:events.append((stage,identity)))
    assert [stage for stage,_ in events]==['checkpoint-public-download-verified','request-public-commit-verified',
        'actions-anchor-signature-verified','anchor-public-download-verified']
    assert events[0][1]['kind']=='registration-packet'


@pytest.mark.parametrize('damage',['registration','source-policy','packet-parent','changed-deadline','wrong-bundle','changed-actions','corrupt-public'])
def test_changed_evidence_never_produces_registration_receipt(tmp_path,monkeypatch,damage):
    packet,r,source,provider,calls=configured(tmp_path,monkeypatch);out=tmp_path/'publisher';deadline=int(time.time())+300
    expected=digest(r)
    if damage in ('changed-deadline','wrong-bundle','changed-actions','corrupt-public'):
        original=m.actions_artifact
        def interrupt(*a,**kw):raise EvidenceError('injected interruption after public packet')
        monkeypatch.setattr(m,'actions_artifact',interrupt)
        with pytest.raises(EvidenceError,match='interruption'):m.publish(packet,expected,source,tmp_path,out,deadline)
        monkeypatch.setattr(m,'actions_artifact',original)
        assert provider.commits==1
        if damage=='changed-deadline':deadline+=1
        elif damage=='wrong-bundle':
            def wrong(rev,dest,limit,**kw):
                receipt=original(rev,dest,limit,**kw);write_json(dest/'registration.sigstore.json',{'attacker-policy':'not selected'});return receipt
            monkeypatch.setattr(m,'actions_artifact',wrong)
        elif damage=='changed-actions':
            receipt=original('2'*40,out/'actions',deadline);receipt['attempt']=2;write_json(out/'actions.json',receipt)
        else:
            monkeypatch.setattr(shared.transport,'download',lambda *a:(_ for _ in ()).throw(EvidenceError('actual public bytes corrupted')))
    elif damage=='registration':expected='0'*64
    elif damage=='source-policy':source=PublisherPolicy(**{**asdict(source),'owner_id':'123'})
    else:
        v=read_json(packet/'initial-verification.json');v['prover_tensors_loaded_as_state']=True;write_json(packet/'initial-verification.json',v)
    with pytest.raises(EvidenceError):m.publish(packet,expected,source,tmp_path,out,deadline)
    assert not(out/'verified-registration.json').exists() and provider.commits<=1


def test_uncertain_real_git_push_is_adopted_without_a_second_mutation(tmp_path,monkeypatch):
    f=tmp_path/'fixture';f.mkdir();packet,req=request(f);r=read_json(packet/'registration.json')
    remote=tmp_path/'bare';work=tmp_path/'seed';work.mkdir()
    def git(*args,cwd=work):return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.DEVNULL).decode().strip()
    git('init','--bare',str(remote));git('init','-b',m.BRANCH);(work/'README').write_text('synthetic fixture')
    git('add','README');git('-c','user.name=Test','-c','user.email=test@example.org','commit','-m','initial')
    git('remote','add','origin',str(remote));git('push','origin',m.BRANCH)
    monkeypatch.setattr(m,'REMOTE',str(remote));monkeypatch.setattr(m,'verify_code',lambda *a:{'explicit-source-double':True});pushes=[]
    def execute(args,**kw):
        if 'commit' in args or args[:2]==['git','push']:assert kw['timeout']==600
        result=m.command(args,**kw)
        if args[:2]==['git','push']:pushes.append(args);raise EvidenceError('lost successful push response')
        return result
    with pytest.raises(EvidenceError,match='lost'):m.request_commit(req,r,tmp_path/'publication',execute=execute)
    revision=m.request_commit(req,r,tmp_path/'publication',execute=execute)
    assert len(pushes)==1 and '--force' not in pushes[0] and git('rev-parse',m.BRANCH,cwd=remote)==revision
    name=f"project/production-commitments/{r['run_id']}-{r['attempt_id']}.json"
    assert git('diff-tree','--no-commit-id','--name-status','-r',revision,cwd=remote)=='A\t'+name


@pytest.mark.parametrize('damage',[None,'workflow','rerun','head','failed','ambiguous','expired'])
def test_actions_checks_exact_registration_workflow_and_single_attempt(tmp_path,damage):
    calls=[]
    def execute(args,**kw):
        calls.append(args)
        if args[1:3]==['run','list']:
            item={'databaseId':19,'headSha':'1'*40,'status':'completed','conclusion':'failure' if damage=='failed' else 'success'}
            if damage=='head':item['headSha']='2'*40
            return json.dumps([item,item] if damage=='ambiguous' else [item])
        if args[1]=='api':
            return json.dumps({'id':19,'head_sha':'1'*40,'head_branch':m.BRANCH,'path':m.PRODUCTION_WORKFLOW if damage!='workflow' else 'wrong.yml',
                'event':'push','run_attempt':2 if damage=='rerun' else 1,'status':'completed','conclusion':'success'})
        assert damage is None and args[:3]==['gh','run','download']
        assert args[args.index('--name')+1]=='pipeline-production-registration-'+'1'*40+'-1'
        Path(args[-1]).mkdir();return ''
    deadline=int(time.time())+(-1 if damage=='expired' else 100)
    if damage:
        with pytest.raises(EvidenceError):m.actions_artifact('1'*40,tmp_path/'out',deadline,execute=execute)
        assert not(tmp_path/'out').exists()
    else:assert m.actions_artifact('1'*40,tmp_path/'out',deadline,execute=execute)['run_id']==19


@pytest.mark.parametrize('elapsed',[300,901])
def test_registration_slow_commit_obeys_original_dispatch_deadline(tmp_path,monkeypatch,elapsed):
    packet,r,source,provider,calls=configured(tmp_path,monkeypatch)
    wall=int(time.time());now=[float(wall),100.];limits=[]
    monkeypatch.setattr(m.time,'time',lambda:now[0])
    monkeypatch.setattr(m.time,'monotonic',lambda:now[1])
    def slow(args,**kw):
        limits.append(kw['timeout']);now[0]+=elapsed;now[1]+=elapsed;return 'synthetic command result'
    monkeypatch.setattr(m,'command',slow)
    original=m.request_commit
    def request_with_hook(req,reg,out,**kw):
        kw['execute'](['git','commit'],timeout=600)
        return original(req,reg,out,**kw)
    monkeypatch.setattr(m,'request_commit',request_with_hook)
    out=tmp_path/'publisher'
    if elapsed>900:
        with pytest.raises(EvidenceError,match='original publication deadline'):
            m.publish(packet,digest(r),source,tmp_path,out,wall+900)
        assert not(out/'verified-registration.json').exists()
    else:
        assert m.publish(packet,digest(r),source,tmp_path,out,wall+900)['result']=='PASS'
    assert limits==[600]
