"""Production stage integration with actual CPU states and local SSH substitutes.

The fixture's already exited toy process is not production training. Publisher
polling is explicitly doubled here; actual signature/byte handoffs have separate
boundary-publisher tests. No provider or production acceptance credit.
"""
from pathlib import Path
import shutil
import time
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
import pytest
import run_production_stage as m
import pod_job_worker
from production_health import ProductionHealth
from production_checkpoint_poll import CheckpointRetention
from production_boundary_poll import BoundaryPublisher
from test_production_health import bound
from test_workload_stage import intent
from test_pipeline import prepared
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal


def configured(prepared,tmp_path):
    control,t,job,root,worker,r,bindings=bound(prepared,tmp_path)
    out=tmp_path/'stage';out.mkdir();shutil.copytree(tmp_path/'launch',out/'launch')
    return control,t,job,root,worker,r,bindings,out


def hooks(tmp_path,t,root,r,h):
    health_file=tmp_path/'health.json'
    checkpoint=CheckpointRetention(r,root,h,health_file,t,tmp_path/'objects',tmp_path/'live',
        {'schema':'ovl.production-checkpoint-copy-policy.v1','export_seconds':60,'maximum_checkpoint_bytes':1024**2})
    publisher=BoundaryPublisher(r,root,h,health_file,t,tmp_path/'publication',
        {k:'explicit-test-only-unused-parent' for k in ('packet','registration-bundle','production-policy','source-policy','source-checkout','config')},
        {'schema':'ovl.production-boundary-publication-policy.v1','boundary_seconds':180,'snapshot_seconds':30,
         'delivery_seconds':30,'python':str(Path(__import__('sys').executable).resolve())})
    return checkpoint,publisher


def execute(tmp_path,data,h,checkpoint,publisher,**kw):
    control,t,job,root,worker,r,bindings,out=data
    return m.run_stage(control,h,job,root,Path(pod_job_worker.__file__),worker,out,tmp_path/'health.json',
        tmp_path/'stop.json','d'*64,checkpoint,publisher,export_seconds=60,sleep=lambda n:time.sleep(.01),**kw)


def test_adopts_original_launch_retains_every_output_then_recovers_without_remote_calls(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    def forbidden(*a,**k):raise AssertionError('must not reissue launch')
    monkeypatch.setattr(m,'launch',forbidden)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        result=execute(tmp_path,data,h,cp,pub)
        assert result['terminal']['exit_code']==0 and h.jobs[root]['finished'] and not h.complete
        proof=read_json(Path(result['retention_path']));assert len(proof['roots'])==2
        receipt=read_json(Path(proof['roots'][1]['receipt_path']))
        assert any(x['path']=='recovery-partial/state.safetensors' for x in receipt['files'])
        monkeypatch.setattr(m,'reconcile_launch',forbidden)
        assert execute(tmp_path,data,h,cp,pub)==result
        state=next(x['path'] for x in receipt['files'] if x['path'].endswith('/state.safetensors'))
        p=Path(receipt['files_directory'])/state;p.chmod(0o600);p.write_bytes(b'altered retained state')
        with pytest.raises(EvidenceError):execute(tmp_path,data,h,cp,pub)


def test_live_loop_uses_actual_checkpoint_hook_and_selected_publisher(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    original=m.bounded_read;calls=[]
    def running_once(kind,*args,**kw):
        value=original(kind,*args,**kw)
        if kind=='supervision' and not calls:
            calls.append('observed')
            return {**value,'state':'RUNNING_UNVERIFIED','terminal':None,'runner_alive':True}
        return value
    monkeypatch.setattr(m,'bounded_read',running_once)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        pub.poll=lambda:calls.append('explicit-publisher-double')
        execute(tmp_path,data,h,cp,pub)
        assert calls==['observed','explicit-publisher-double']
        assert list((tmp_path/'live/checkpoints').glob('*/retained/retention.json'))


def test_controller_stop_goes_to_numerical_record_before_hard_worker_stop(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    write_json(tmp_path/'stop.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
        'pod_id':control.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['test stop']})
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        execute(tmp_path,data,h,cp,pub)
        assert read_json(tmp_path/'record/remote/request-stop')['reasons']==['test stop']
        assert not(tmp_path/'control/remote/jobs'/root/'request-stop').exists()


@pytest.mark.parametrize('damage',['worker','checkpoint-job','publisher-job','foreign-stop','reserve','grace-reserve'])
def test_invalid_bindings_or_stop_fail_without_completion(prepared,tmp_path,damage):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        if damage=='worker':bindings[root]['worker_sha256']='e'*64
        elif damage=='checkpoint-job':cp.job='e'*64
        elif damage=='publisher-job':pub.job='e'*64
        elif damage=='reserve':h.plan['provider_terminate_epoch']=read_json(job)['deadline_epoch']+60
        elif damage=='grace-reserve':h.plan['input']['checkpoint_grace_seconds']=60
        else:write_json(tmp_path/'stop.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':'e'*64,
            'pod_id':control.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['foreign']})
        with pytest.raises(EvidenceError):execute(tmp_path,data,h,cp,pub)
        assert not h.complete and not(out/'stage-result.json').exists()


def test_interrupted_terminal_copy_preserves_bytes_and_original_deadline(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    original=m.retain;limits=[]
    def fail(*args,**kw):
        limits.append(args[7]);p=Path(args[6]);p.mkdir();(p/'sole.partial').write_bytes(b'preserve')
        raise EvidenceError('explicit interrupted terminal transfer')
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        monkeypatch.setattr(m,'retain',fail)
        with pytest.raises(EvidenceError,match='interrupted'):execute(tmp_path,data,h,cp,pub)
        before=read_json(out/'terminal-export-intent.json')
        def retry(*a,**kw):limits.append(a[7]);return original(*a,**kw)
        monkeypatch.setattr(m,'retain',retry);result=execute(tmp_path,data,h,cp,pub)
        assert limits[0]==limits[1] and before==read_json(out/'terminal-export-intent.json')
        assert (out/'terminal-export/sole.partial').read_bytes()==b'preserve'
        assert 'terminal-export-retry' in result['retention_path']


def test_expired_copy_window_is_not_renewed_on_adoption(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    def fail(*args,**kw):
        p=Path(args[6]);p.mkdir();(p/'partial').write_bytes(b'preserved')
        raise EvidenceError('injected transfer interruption')
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        monkeypatch.setattr(m,'retain',fail)
        with pytest.raises(EvidenceError,match='interruption'):execute(tmp_path,data,h,cp,pub)
        before=read_json(out/'terminal-export-intent.json')
        h.wall=lambda:before['deadline_epoch']
        with pytest.raises(EvidenceError,match='original terminal export deadline expired'):execute(tmp_path,data,h,cp,pub)
        assert read_json(out/'terminal-export-intent.json')==before
        assert not(out/'terminal-export-retry').exists() and not h.complete


def test_missing_publisher_is_rejected_before_new_launch(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        with pytest.raises(EvidenceError,match='public boundary hook'):execute(tmp_path,data,h,cp,None)
        assert not h.jobs


def test_full_replay_retains_all_outputs_and_stops_worker_without_publication(prepared,tmp_path):
    data=configured(prepared,tmp_path);control,t,job,old,worker,r,_,out=data
    value=read_json(job);value['kind']='full-replay';write_json(job,value);root=digest(value)
    # Distinct local toy process and fence; never a production/CUDA replay claim.
    replay_out=tmp_path/'replay-stage'
    m.launch(control,job,root,Path(pod_job_worker.__file__),worker,replay_out/'launch',int(time.time())+30)
    from test_pod_job_worker import exited
    assert exited(tmp_path/'control/remote/jobs'/root)['exit_code']==0
    bindings={root:{'job_file':job,'worker_sha256':worker,'control':control,'transports':[t]}}
    data=(control,t,job,root,worker,r,bindings,replay_out)
    write_json(tmp_path/'stop.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
        'pod_id':control.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['replay stop test']})
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings)
        cp=CheckpointRetention(r,root,h,tmp_path/'health.json',t,tmp_path/'objects',tmp_path/'replay-live',
            {'schema':'ovl.production-checkpoint-copy-policy.v1','export_seconds':60,'maximum_checkpoint_bytes':1024**2},
            envelopes=read_json(tmp_path/'record/remote/chain.json')['boundaries'])
        result=execute(tmp_path,data,h,cp,None)
        assert result['terminal']['exit_code']==0 and h.jobs[root]['finished'] and not h.complete
        assert (replay_out/'worker-stop-delivery.json').exists()
        assert not(tmp_path/'record/remote/request-stop').exists()


def test_production_adoption_control_allowance_preserves_original_deadlines(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    now=int(time.time());w=intent();original_job=job.read_bytes();original_plan=digest(w['plan']);seen=[]
    class Selected(Exception):pass
    def intercept(*args,**kw):seen.append(args[-1]);raise Selected()
    monkeypatch.setattr(m,'reconcile_launch',intercept)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings);h.wall=lambda:now
        cp,pub=hooks(tmp_path,t,root,r,h)
        with pytest.raises(Selected):execute(tmp_path,data,h,cp,pub)
    assert seen==[min(w['plan']['external_terminate_epoch'],now+120)]
    assert job.read_bytes()==original_job and digest(w['plan'])==original_plan


@pytest.mark.parametrize('stop_kind',['controller','graceful'])
def test_production_stop_during_uploads_prevents_new_start(prepared,tmp_path,stop_kind):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    # The fixture already retains a completed toy run. Preserve its fence while
    # exercising only a prospective dispatch, stopped before any start command.
    (out/'launch').rename(out/'prior-launch');now=int(time.time());clock=[now];uploads=[]
    original=control.put;original_stream=control.stream;starts=[]
    def stream(argv,*args,**kw):
        if 'start' in argv:starts.append(argv);raise AssertionError('late production start')
        return original_stream(argv,*args,**kw)
    control.stream=stream
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);h.wall=lambda:clock[0]
        if stop_kind=='graceful':h.plan['request_checkpoint_epoch']=now+70
        cp,pub=hooks(tmp_path,t,root,r,h);original_job=job.read_bytes();original_plan=digest(h.plan);progress=h.progress
        def delayed(name,*args,**kwargs):
            receipt=original(name,*args,**kwargs);uploads.append((name,args[-1]));clock[0]+=40
            if len(uploads)==2 and stop_kind=='controller':
                write_json(tmp_path/'stop.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
                    'pod_id':h.pod,'observed_epoch':clock[0],'reasons':['synthetic stop during upload']})
            return receipt
        control.put=delayed
        with pytest.raises(EvidenceError,match='before production launch fence'):execute(tmp_path,data,h,cp,pub)
        assert len(uploads)==2 and not starts and not(out/'launch/launch-intent.json').exists()
        assert {deadline for name,deadline in uploads}=={min(read_json(job)['deadline_epoch'],h.plan['request_checkpoint_epoch'],now+120)}
        assert job.read_bytes()==original_job and digest(h.plan)==original_plan and h.progress==progress and not h.complete


def test_pending_production_stop_precedes_failing_adoption(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    stop=tmp_path/'stop.json';write_json(stop,{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
        'pod_id':control.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['synthetic pending stop']})
    def blocked(*args):
        assert (tmp_path/'record/remote/request-stop').read_bytes()==stop.read_bytes()
        raise EvidenceError('synthetic adoption unavailable')
    monkeypatch.setattr(m,'reconcile_launch',blocked)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        with pytest.raises(EvidenceError,match='adoption unavailable'):execute(tmp_path,data,h,cp,pub)
        assert not h.complete and (out/'stop-delivery.json').exists()


def test_missing_early_controller_stop_cannot_resume_publication(prepared,tmp_path,monkeypatch):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    stop=tmp_path/'stop.json';write_json(stop,{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
        'pod_id':control.profile['pod_id'],'observed_epoch':int(time.time()),'reasons':['synthetic early stop']})
    original=m.reconcile_launch
    def remove(*args):
        result=original(*args);stop.unlink();return result
    monkeypatch.setattr(m,'reconcile_launch',remove)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);cp,pub=hooks(tmp_path,t,root,r,h)
        pub.poll=lambda:pytest.fail('publication resumed after missing stop')
        with pytest.raises(EvidenceError,match='disappeared'):execute(tmp_path,data,h,cp,pub)
        assert not h.complete and not(out/'stage-result.json').exists()


@pytest.mark.parametrize('crossing',['numerical-delivery','adoption'])
def test_stop_clock_rechecked_after_blocking_control_operation(prepared,tmp_path,monkeypatch,crossing):
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    now=int(time.time());clock=[now];hard=read_json(job)['deadline_epoch'];events=[]
    write_json(tmp_path/'stop.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,
        'pod_id':control.profile['pod_id'],'observed_epoch':now,'reasons':['synthetic clock crossing']})
    original_put=t.put
    def put(*args,**kw):
        result=original_put(*args,**kw);events.append('numerical')
        if crossing=='numerical-delivery':clock[0]=hard
        return result
    t.put=put
    def adopt(*args):
        events.append('adoption')
        if crossing=='adoption':
            assert args[-1]<=hard;clock[0]=hard
        else:assert (tmp_path/'control/remote/jobs'/root/'request-stop').exists()
    monkeypatch.setattr(m,'reconcile_launch',adopt)
    def inspect(*args,**kw):
        assert (tmp_path/'control/remote/jobs'/root/'request-stop').exists()
        raise EvidenceError('synthetic inspected stop ordering')
    monkeypatch.setattr(m,'bounded_read',inspect)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings);h.wall=lambda:clock[0]
        cp,pub=hooks(tmp_path,t,root,r,h)
        with pytest.raises(EvidenceError,match='inspected stop ordering'):execute(tmp_path,data,h,cp,pub)
        assert events==['numerical','adoption']
        assert read_json(out/'stop-intent.json')['hard_stop_epoch']==hard
