"""Off-pod handoff with actual safe checkpoints and explicit remote/identity doubles."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publish_progress_boundary as m
from test_pipeline import prepared
from test_production_chain import actual_artifacts
from test_production_commitment import request as production_request
from test_progress_commitment import request as progress_request
from test_evidence_publication import Fake
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.anchoring import PublisherPolicy
from ovl_pipeline.production_anchoring import PACKET_FILES
from ovl_pipeline.production_identity import ProductionPublisherPolicy,PRODUCTION_WORKFLOW
from ovl_pipeline import progress_anchoring as pa


def setup(prepared,tmp_path,monkeypatch,damage=None,*,persistent=False):
    numerical=tmp_path/'numerical';numerical.mkdir();r,root,envs,key,chain,streams=actual_artifacts(prepared,numerical)
    source=tmp_path/'source';source.mkdir();packet,rr=production_request(source)
    write_json(packet/'registration.json',r);rr['registration_sha256']=root
    rr['packet']['prefix']='production-registration/'+root;rr['packet']['inventory']=inventory(packet,sorted(PACKET_FILES))
    sp=PublisherPolicy(**rr['source_policy']);pp=ProductionPublisherPolicy(**{**asdict(sp),'workflow':PRODUCTION_WORKFLOW,'statement_sha256':root})
    bundle=tmp_path/'bundle';bundle.mkdir();write_json(bundle/'registration.sigstore.json',{'explicit-test-double':True})
    config={'schema':'ovl.progress-dispatch.v1','registration_request':rr,
            'registration_anchor':{'repo':rr['packet']['repo'],'revision':'1'*40,'prefix':'production-anchors/'+root,
                                   'inventory':inventory(bundle,['registration.sigstore.json'])}}
    monkeypatch.setattr(m,'verify_packet',lambda *a,**k:{'explicit-publisher-test-double':True})
    def anchor(statement,bundle,policy,**kw):
        if damage=='signature':raise EvidenceError('signature rejected by explicit publisher double')
        assert file_root(statement)==policy.statement_sha256
        return {'explicit-publisher-test-double':True,'statement_sha256':policy.statement_sha256}
    monkeypatch.setattr(pa,'verify_anchor',anchor)
    provider=Fake();upload=m.transport.upload;download=m.transport.download;reconcile=m.transport.reconcile
    # These tests replace the remote provider. Keep the original deadline flowing
    # through the real uploader; exercise the privacy gate separately with its
    # explicit adversarial/installed-scanner tests, never a real publication here.
    import publication_export_gate
    monkeypatch.setattr(publication_export_gate,'require_review',
        lambda plan_path,*a,**kw:{'synthetic-test-double':True,'plan_sha256':digest(read_json(plan_path))})
    monkeypatch.setattr(m.transport,'upload',lambda *a,**kw:upload(*a,api=provider,**kw))
    monkeypatch.setattr(m.transport,'download',lambda *a:download(*a,api=provider,fetch_file=provider.fetch))
    monkeypatch.setattr(m.transport,'reconcile',lambda *a:reconcile(*a,api=provider))
    committed={}
    def request(value,registration,directory,**kwargs):
        if damage=='failed-push':raise EvidenceError('push failure')
        n=len(value['envelopes']);revision=str(n)*40
        if n in committed:assert committed[n]==digest(value)
        committed[n]=digest(value);return revision
    monkeypatch.setattr(m,'request_commit',request)
    fail_once=[damage=='restart-after-upload']
    def actions(revision,out,deadline,**kwargs):
        if fail_once[0]:fail_once[0]=False;raise EvidenceError('interrupted after upload')
        value=read_json(out.parent/'expected-statement.json')
        if damage=='wrong-statement':value['boundary_sha256']='a'*64
        current=out/f'progress/progress-{value["index"]:05d}';current.mkdir(parents=True)
        write_json(current/'statement.json',value);write_json(current/'statement.sigstore.json',{'explicit-test-double':True})
        return {'explicit-actions-test-double':True,'revision':revision}
    monkeypatch.setattr(m,'actions_artifact',actions)
    deadline=int(time.time())+600
    def run(index,previous=None,policies=None):
        selected=envs[:index+1];body=selected[-1]['body']
        write_json(chain/'chain.json',{'schema':'ovl.production-chain.v1','complete':False,'boundaries':selected})
        waiting={'schema':'ovl.awaiting-public-progress.v1','registration_sha256':root,'index':index,
                 'boundary_sha256':digest(selected[-1]),'checkpoint_path':body['checkpoint_path'],'checkpoint':body['checkpoint']}
        if damage=='wrong-waiting':waiting['boundary_sha256']='0'*64
        write_json(chain/'awaiting-anchor.json',waiting)
        if damage=='checkpoint':(chain/body['checkpoint_path']/'state.safetensors').write_bytes(b'changed')
        if persistent:
            import persistent_publication as service
            # The service hashes complete immutable input trees. Keep its mutable
            # output/state outside them, including the independently selected source.
            inputs=tmp_path/f'service-inputs-{index:05d}';inputs.mkdir()
            source=inputs/'source';source.mkdir()
            previous=previous or inputs/'empty-prefix'
            previous.mkdir(exist_ok=True)
            for name,value in [('production-policy',asdict(pp)),('source-policy',asdict(sp)),
                               ('config',config),('previous-policies',[asdict(p) for p in (policies or [])])]:
                write_json(inputs/(name+'.json'),value)
            args={'packet':str(packet),'registration-bundle':str(bundle/'registration.sigstore.json'),
                  'source-checkout':str(source),'chain-directory':str(chain),'previous-directory':str(previous),
                  'output':str(tmp_path/f'publication/boundary-{index:05d}'),
                  **{name:str(inputs/(name+'.json')) for name in ('production-policy','source-policy','config','previous-policies')}}
            spec=service.selection(root,digest(selected[-1]),deadline,args,python=Path(sys.executable).resolve())
            if damage=='service-boundary':spec['boundary_sha256']='c'*64
            if damage=='service-input-change':
                original=m.publish
                def changed(*a,**kw):
                    result=original(*a,**kw)
                    (source/'unselected.py').write_text('# changed during publication')
                    return result
                monkeypatch.setattr(m,'publish',changed)
            state=tmp_path/f'service-state-{index:05d}';state.mkdir();write_json(state/'selection.json',spec)
            result=service.worker(spec,digest(spec),state)
            assert read_json(state/'result.json')['ack_sha256']==digest(result)
            with pytest.raises(FileExistsError):service.worker(spec,digest(spec),state)
            return result
        return m.publish(packet,bundle/'registration.sigstore.json',pp,sp,tmp_path,config,chain,
                         previous or tmp_path/'none',policies or [],tmp_path/f'publication/boundary-{index:05d}',deadline)
    return run,provider,committed


def file_root(path):
    from ovl_pipeline.canonical import file_hash
    return file_hash(path)


@pytest.mark.parametrize('persistent',[False,True])
def test_real_checkpoint_publication_precedes_verified_ack_and_links_next(prepared,tmp_path,monkeypatch,persistent):
    run,provider,committed=setup(prepared,tmp_path,monkeypatch,persistent=persistent)
    first=run(0);assert provider.commits==2 and first['checkpoint_download']['result']=='PASS'
    assert first['training_replay']=='NOT_RUN'
    second=run(1,Path(first['anchor_directory']),[pa.ProgressPublisherPolicy(**first['policy'])])
    assert provider.commits==4 and second['index']==1
    assert second['public_prefix_check']['anchors'][0]['statement_sha256']==first['policy']['statement_sha256']
    if not persistent:
        with pytest.raises(EvidenceError,match='already exists'):run(0)
    assert provider.commits==4


@pytest.mark.parametrize('damage',['wrong-waiting','checkpoint','wrong-statement','signature','failed-push'])
@pytest.mark.parametrize('persistent',[False,True])
def test_missing_or_changed_evidence_never_acknowledges(prepared,tmp_path,monkeypatch,damage,persistent):
    run,provider,committed=setup(prepared,tmp_path,monkeypatch,damage,persistent=persistent)
    with pytest.raises(EvidenceError):run(0)
    assert not(tmp_path/'publication/boundary-00000/ack.json').exists()
    if damage in ('wrong-waiting','checkpoint'):assert provider.commits==0


@pytest.mark.parametrize('damage',['service-boundary','service-input-change'])
def test_service_never_returns_result_for_changed_selected_identity(prepared,tmp_path,monkeypatch,damage):
    run,provider,_=setup(prepared,tmp_path,monkeypatch,damage,persistent=True)
    with pytest.raises(EvidenceError):run(0)
    assert not(tmp_path/'service-state-00000/result.json').exists()
    assert (tmp_path/'service-state-00000/worker-started.json').exists()


def test_restart_pins_original_archive_and_never_republishes_checkpoint(prepared,tmp_path,monkeypatch):
    run,provider,committed=setup(prepared,tmp_path,monkeypatch,'restart-after-upload')
    with pytest.raises(EvidenceError,match='interrupted'):run(0)
    assert provider.commits==1
    provider.sha='9'*40 # unrelated public head advancement must not change request
    ack=run(0)
    assert ack['checkpoint_archive']['revision']=='3'*40 and provider.commits==2 and len(committed)==1


@pytest.mark.parametrize('mode',['fail','rerun','ambiguous','wrong-head','expired'])
def test_actions_identity_failures_and_expired_wait_never_download(tmp_path,mode):
    def execute(args,**kw):
        if args[1]=='api':return json.dumps({'id':1,'head_sha':'1'*40,'head_branch':m.BRANCH,'path':m.PROGRESS_WORKFLOW,
                                            'event':'push','run_attempt':2,'status':'completed','conclusion':'success'})
        assert args[1:3]==['run','list'] and 'attempt' not in args[-1]
        item={'databaseId':1,'headSha':'1'*40,'status':'completed','conclusion':'success','attempt':1}
        if mode=='fail':item['conclusion']='failure'
        elif mode=='rerun':item['attempt']=2
        elif mode=='wrong-head':item['headSha']='2'*40
        return json.dumps([item,item] if mode=='ambiguous' else [item])
    with pytest.raises(EvidenceError):m.actions_artifact('1'*40,tmp_path/'out',int(time.time())+(-1 if mode=='expired' else 100),execute=execute)
    assert not(tmp_path/'out').exists()


def test_uncertain_real_git_push_reconciles_without_second_push(tmp_path,monkeypatch):
    fixture=tmp_path/'fixture';fixture.mkdir();request=progress_request(fixture)
    remote=tmp_path/'bare';work=tmp_path/'seed';work.mkdir()
    def git(*args,cwd=work):return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.DEVNULL).decode().strip()
    git('init','--bare',str(remote));git('init','-b',m.BRANCH)
    (work/'README').write_text('synthetic public repository')
    git('add','README');git('-c','user.name=Test','-c','user.email=test@example.org','commit','-m','initial')
    git('remote','add','origin',str(remote));git('push','origin',m.BRANCH)
    monkeypatch.setattr(m,'REMOTE',str(remote));monkeypatch.setattr(m,'verify_code',lambda *a:{'explicit-source-test-double':True})
    pushes=[]
    def execute(args,**kwargs):
        if 'commit' in args or args[:2]==['git','push']:assert kwargs['timeout']==600
        result=m.command(args,**kwargs)
        if args[:2]==['git','push']:
            pushes.append(args);raise EvidenceError('client lost successful push response')
        return result
    directory=tmp_path/'publisher';r={'run_id':'test-run','attempt_id':'attempt'}
    with pytest.raises(EvidenceError,match='lost'):m.request_commit(request,r,directory,execute=execute)
    revision=m.request_commit(request,r,directory,execute=execute)
    assert len(pushes)==1 and len(revision)==40
    assert git('rev-parse',m.BRANCH,cwd=remote)==revision
    assert '--force' not in pushes[0]


def test_actions_wait_binds_exact_workflow_and_artifact_name(tmp_path):
    calls=[]
    def execute(args,**kwargs):
        calls.append(args)
        if args[1:3]==['run','list']:
            return json.dumps([{'databaseId':19,'headSha':'1'*40,'status':'completed','conclusion':'success'}])
        if args[1]=='api':
            return json.dumps({'id':19,'head_sha':'1'*40,'head_branch':m.BRANCH,'path':m.PROGRESS_WORKFLOW,
                               'event':'push','run_attempt':1,'status':'completed','conclusion':'success'})
        assert args[:3]==['gh','run','download'] and args[3]=='19'
        assert args[args.index('--name')+1]=='pipeline-production-progress-'+'1'*40+'-1'
        Path(args[-1]).mkdir();return ''
    receipt=m.actions_artifact('1'*40,tmp_path/'out',int(time.time())+100,execute=execute)
    assert receipt['run_id']==19 and len(calls)==3


def test_anchor_copy_rejects_symlink_before_read(tmp_path):
    source=tmp_path/'source';source.mkdir();(tmp_path/'private').write_text('do not copy')
    (source/'statement.json').symlink_to(tmp_path/'private');write_json(source/'statement.sigstore.json',{})
    with pytest.raises(EvidenceError):m.copy_anchor(source,tmp_path/'out')
    assert not(tmp_path/'out').exists()


@pytest.mark.parametrize('elapsed',[300,601])
def test_progress_slow_commit_obeys_original_dispatch_deadline(prepared,tmp_path,monkeypatch,elapsed):
    run,_,_=setup(prepared,tmp_path,monkeypatch)
    now=[time.time(),100.];limits=[]
    monkeypatch.setattr(m.time,'time',lambda:now[0])
    monkeypatch.setattr(m.time,'monotonic',lambda:now[1])
    def slow(args,**kw):
        limits.append(kw['timeout']);now[0]+=elapsed;now[1]+=elapsed;return 'synthetic command result'
    factory=m.deadline_command
    monkeypatch.setattr(m,'deadline_command',lambda deadline:factory(deadline,execute=slow))
    original=m.request_commit
    def request_with_hook(value,registration,directory,**kw):
        kw['execute'](['git','commit'],timeout=600)
        return original(value,registration,directory,**kw)
    monkeypatch.setattr(m,'request_commit',request_with_hook)
    if elapsed>600:
        with pytest.raises(EvidenceError,match='original publication deadline'):run(0)
        assert not list((tmp_path/'publication').glob('**/ack.json'))
    else:
        assert run(0)['schema']=='ovl.verified-public-progress-ack.v1'
    assert len(limits)==1 and 599<limits[0]<=600
