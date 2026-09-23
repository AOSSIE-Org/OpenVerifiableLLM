"""Real setup, byte reads, SSH metadata, stage export and health-file consumption.

Worker launch/supervision, HTTPS, installation and time are explicit doubles.
The coordinator polls independently of the blocked synthetic network reader.
"""
from datetime import datetime,timezone
from pathlib import Path
import json,queue,threading,time,sys,os
import pytest
import pod_fetch_runtime as fetcher
import pod_public_setup as setupper
import pod_observation_retry as polling
import run_workload_stage as stage
from run_rental_controller import normalized
from sustained_health import SustainedHealth
from test_pod_job_client import fixture as transport_fixture
from test_pod_public_setup import fixture as setup_fixture
from test_pod_fetch_runtime import Response
from test_external_watchdog import intent
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,write_json
from ovl_pipeline.supervision import Journal,rental_plan,observe


@pytest.mark.parametrize('connected,slow_offline,report_offline',[(False,False,False),(True,False,True),(False,True,False),(True,True,True),(True,True,False)])
def test_descriptor_polling_exports_and_controller_stall_decision(tmp_path,monkeypatch,connected,slow_offline,report_offline):
    transport,remote,calls,job_file,_,worker,worker_hash=transport_fixture(tmp_path)
    inputs,config,cfg=setup_fixture(remote);output=remote/'setup-evidence'
    data=b'x'*(8*1024**2);wheel={'path':'synthetic-1-py3-none-any.whl','bytes':len(data),
        'sha256':__import__('hashlib').sha256(data).hexdigest(),'url':'https://files.pythonhosted.org/synthetic.whl'}
    write_json(inputs/'plan.json',{'schema':'ovl.public-wheel-download.v1','files':[wheel]})
    cfg.update(wheel_plan_sha256=file_hash(inputs/'plan.json'),download_seconds=600);write_json(config,cfg)
    # The local SSH double maps the selected virtual remote root to this directory.
    if connected:monkeypatch.setenv('OVL_ACTIVITY_FILE',str(output/'activity.json'))
    else:monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    now=int(time.time());clock=[now];origin=now
    w=intent();w['plan']=rental_plan({**w['plan']['input'],'now_epoch':now,'maximum_seconds':3600})
    w['payload']['terminateAfter']=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    w['creation_latest_epoch']=now+30;w['baseline']['balance_usd']='100'
    w['baseline'].update(observed_epoch=now,http_clock={'server_epoch':now,'request_started_epoch':now,'request_completed_epoch':now})
    value=read_json(job_file);value.update(kind='setup',deadline_epoch=now+900,
        argv=[str(Path(sys.executable).resolve()),'-I','-S',str(Path(setupper.__file__).resolve()),
              '--config',str(config),'--config-sha256',file_hash(config),'--inputs',str(inputs),
              '--runtime',str(remote/'runtime'),'--output',str(output),'--deadline',str(now+900)],
        environment={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8',
                     **({'OVL_ACTIVITY_FILE':transport.profile['remote_root']+'/setup-evidence/activity.json'} if connected else {})},
        export_roots=[transport.profile['remote_root']+'/setup-evidence'])
    selected_files=[Path(sys.executable).resolve(),Path(setupper.__file__).resolve(),Path(fetcher.__file__).resolve(),
                    *sorted(inputs.iterdir())]
    value['required_files']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in selected_files]
    write_json(job_file,value);root=digest(value);binding={'schema':'ovl.public-input-binding.v1',
        'plan_sha256':cfg['wheel_plan_sha256'],'bytes':len(data)}
    # A foreign same-plan snapshot is outside the selected polling path.
    foreign=remote/'other-job';foreign.mkdir();write_json(foreign/'activity.json',{
        'schema':'ovl.public-input-transfer.v1','process_instance':'f'*32,'pid':999,
        'plan_sha256':binding['plan_sha256'],'total_bytes':len(data),'received_bytes':len(data),
        'scope':'operator-supervision-only-not-input-verification'})
    ready=threading.Event();grant=queue.Queue();done=threading.Event();phase=['download'];errors=[]
    terminal={'schema':'ovl.workload-job-exit.v1','job_sha256':root,'state':'EXITED','exit_code':0}
    class GatedResponse(Response):
        def read(self,n):
            ready.set();grant.get(timeout=15);return super().read(n)
    class Client:
        def open(self,request,timeout):return GatedResponse(data,request.full_url)
    def execute_child(argv,**kwargs):
        if argv[3].endswith('fetch.py'):
            assert ('OVL_ACTIVITY_FILE' in kwargs['env'])==connected
            fetcher.fetch(inputs/'plan.json',cfg['wheel_plan_sha256'],inputs/'wheels',output/'downloads.json',
                origin+600,opener=Client(),wall=lambda:clock[0],monotonic=lambda:clock[0]-origin)
        else:
            assert ('OVL_ACTIVITY_FILE' in kwargs['env'])==(connected and report_offline);phase[0]='offline'
            if slow_offline:
                import pod_runtime_setup as offline
                report=offline.SetupProgress(clock=lambda:clock[0])
                report('source')
                fdopen=offline.os.fdopen
                class SlowFile:
                    def __init__(self,f):self.f=f
                    def __enter__(self):return self
                    def __exit__(self,*args):return self.f.__exit__(*args)
                    def fileno(self):return self.f.fileno()
                    def read(self,n):
                        ready.set();grant.get(timeout=15);return self.f.read(n)
                def selected_fdopen(fd,*args,**kw):
                    f=fdopen(fd,*args,**kw)
                    # Only read-mode archive streams are slowed, never telemetry writes.
                    return SlowFile(f) if args==('rb',) else f
                with monkeypatch.context() as patch:
                    patch.setattr(offline.os,'fdopen',selected_fdopen)
                    offline.cache_wheels(inputs/'wheels',remote/'local-wheel-cache',progress=report)
                report('selection')
            else:
                ready.set();grant.get(timeout=15)
            (output/'offline').mkdir();write_json(output/'offline/setup.json',{
                'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':cfg['offline_config_sha256']})
    def execute(argv,**kwargs):
        # Honor the sanitized child environment even though execution is doubled.
        # The silent case deliberately reproduces the old missing-forwarding path.
        env=dict(kwargs['env'])
        if not argv[3].endswith('fetch.py') and not report_offline:env.pop('OVL_ACTIVITY_FILE',None)
        with monkeypatch.context() as child:
            if 'OVL_ACTIVITY_FILE' in env:child.setenv('OVL_ACTIVITY_FILE',env['OVL_ACTIVITY_FILE'])
            else:child.delenv('OVL_ACTIVITY_FILE',raising=False)
            return execute_child(argv,**{**kwargs,'env':env})
    def work():
        try:
            selected=read_json(job_file);assert digest(selected)==root
            args=dict(zip(selected['argv'][4::2],selected['argv'][5::2]))
            setupper.setup(args['--config'],args['--config-sha256'],args['--inputs'],args['--runtime'],
                           args['--output'],int(args['--deadline']),execute=execute)
        except BaseException as error:errors.append(error);terminal['exit_code']=1
        finally:
            write_json(remote/'jobs'/root/'exit.json',terminal);done.set();ready.set()
    thread=threading.Thread(target=work)
    def launch(*args,**kwargs):
        from pod_job_worker import validate_job
        validate_job(value)
        for f in value['required_files']:assert file_hash(Path(f['path']))==f['sha256']
        assert binding['plan_sha256']==read_json(config)['wheel_plan_sha256']
        assert binding['bytes']==sum(f['bytes'] for f in read_json(inputs/'plan.json')['files'])
        if connected:
            virtual=value['environment']['OVL_ACTIVITY_FILE'];prefix=transport.profile['remote_root']+'/'
            assert virtual.startswith(prefix) and os.environ['OVL_ACTIVITY_FILE']==str(remote/virtual.removeprefix(prefix))
        directory=remote/'jobs'/root;directory.mkdir(parents=True);write_json(directory/'job.json',value)
        write_json(directory/'status.json',{'schema':'ovl.pod-job-status.v1','job_sha256':root,'state':'RUNNING'})
        thread.start()
    def supervision(*args,**kwargs):
        if done.is_set():return {'state':'EXITED','terminal':terminal,'child_alive':False,'runner_alive':False}
        return {'state':'RUNNING'}
    monkeypatch.setattr(stage,'launch',launch);monkeypatch.setattr(polling,'job_supervision',supervision)
    decisions=[];health_file=tmp_path/'health.json';stop=tmp_path/'stop.json';export_calls=[]
    with Journal(tmp_path/'journal').lease() as journal:
        def health():return SustainedHealth(journal,w,transport.profile['pod_id'],{}, {root:binding},
            wall=lambda:clock[0],clock=lambda:{'boot_id':'synthetic-clock','boottime_ms':(clock[0]-origin)*1000})
        h=health();restarted=[]
        def check_controller():
            current=read_json(health_file);pod={'id':h.pod,'gpuCount':1,'costPerHr':'0.3','adjustedCostPerHr':'0.3'}
            account={'observed_epoch':clock[0],'balance_usd':'99','account_hourly_usd':'0.3','pods':[pod]}
            watchdog={'observed_epoch':clock[0],'plan_sha256':digest(w['plan']),
                'external_terminate_epoch':w['plan']['external_terminate_epoch'],'state':'ARMED'}
            decision=observe(w['plan'],normalized(w,account,pod,watchdog,current,clock[0]));decisions.append(decision)
            if decision['action']!='CONTINUE' and not stop.exists():write_json(stop,{
                'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,'pod_id':h.pod,
                'observed_epoch':clock[0],'reasons':decision['reasons']})
        def sleep(seconds):
            assert ready.wait(15),'synthetic producer did not reach its gated read';ready.clear()
            check_controller()
            if clock[0]-origin>=200 and not restarted:
                before=h.progress;adopted=health();assert adopted.progress==before
                h.__dict__.update(adopted.__dict__);restarted.append(True)
            clock[0]+=(45 if slow_offline else 115) if phase[0]=='offline' else 50;grant.put(True)
            assert ready.wait(15),'producer did not complete its granted phase'
        original_export=stage.export_tree
        def export(*args,**kwargs):
            with pytest.raises(EvidenceError):h.start_job({'schema':'ovl.selected-workload-job.v1',
                'job_sha256':'e'*64,'pod_id':h.pod,'kind':'setup'})
            receipt=original_export(*args,**kwargs);export_calls.append(receipt);return receipt
        monkeypatch.setattr(stage,'export_tree',export)
        try:
            result=stage.run_stage(transport,h,job_file,root,worker,worker_hash,tmp_path/'stage',
                health_file,stop,'d'*64,sleep=sleep)
        finally:
            for _ in range(12):grant.put(True)
            thread.join(20)
        assert not thread.is_alive() and not errors and result['exit']['exit_code']==0
        assert clock[0]-origin==(855 if slow_offline else 565) and restarted and len(export_calls)==2
        assert h.jobs[root]['finished'] and not h.complete
        assert read_json(Path(result['exports'][1]['directory'])/'offline/setup.json')['result']=='PASS'
        if connected and (not slow_offline or report_offline):
            assert all(d['action']=='CONTINUE' for d in decisions) and not stop.exists()
            assert h.download_processes[root]['pid']!=999
        else:assert any('stalled-or-future-progress' in d['reasons'] for d in decisions)
        assert not any('other-job' in command[-1] for command in calls)
