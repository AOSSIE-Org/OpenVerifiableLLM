"""Real local process/transfer lifecycle with explicit non-GPU endpoints."""
from pathlib import Path
import time
import pytest

import run_sustained_pilot as m
from sustained_pilot_selection import DEADLINE
from test_workload_coordinator import fixture as finite_fixture
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json,file_hash
from ovl_pipeline.supervision import Journal,ControllerBusy
from test_gpu_pilot import cpu_runtime,prepared


def fixture(tmp_path,*,second=False):
    _,old,rental,t,remote,calls,controller,heartbeat=finite_fixture(tmp_path,second=second)
    stages=[]
    for i,old_stage in enumerate(old['stages']):
        path=tmp_path/old_stage['job_path'];template=read_json(path)
        template['kind']='export';template['deadline_epoch']=DEADLINE
        if i:
            (remote/'output-two').mkdir();(remote/'output-two/data').write_bytes(b'complete second output')
            template['export_roots']=[t.profile['remote_root']+'/output-two']
        write_json(path,template)
        stages.append({'name':old_stage['name'],'template_path':old_stage['job_path'],'template_sha256':digest(template),
                       'work_seconds':20,'export_reserve_seconds':120,'maximum_export_bytes':1024**2,
                       'parent_stage':None,'parent_record_root':None,'parent_binding':None,'validation_binding':None,'download_binding':None,'retention':None})
    plan={**old,'schema':'ovl.sustained-pilot-plan.v1','stages':stages,
          'timing':{k:v for k,v in old['timing'].items() if k!='export_reserve_seconds'}}
    worker=Path(__file__).parents[1]/'scripts/pod_job_worker.py'
    def run():return m.run(plan,digest(plan),rental,controller,heartbeat,t,tmp_path,worker,tmp_path/'sustained',tmp_path/'health.json',sleep=lambda _:time.sleep(.05))
    return run,plan,rental,t,remote,calls,controller,heartbeat


def test_two_real_jobs_complete_and_adopt_without_renewal_or_relaunch(tmp_path):
    run,plan,_,_,_,calls,_,_=fixture(tmp_path,second=True)
    result=run();assert result['outcome']=='EXITED_ZERO' and len(result['stages'])==2
    selected=[read_json(tmp_path/'sustained/derived'/s['name']/'selection.json') for s in plan['stages']]
    assert all(x['deadline_epoch']==x['admitted_epoch']+20 for x in selected)
    before=len(calls);assert run()==result and len(calls)==before
    assert len([c for c in calls if ' start ' in c[-1]])==2 and read_json(tmp_path/'health.json')['complete']


def test_restart_after_complete_stage_never_creates_another_deadline(tmp_path,monkeypatch):
    run,plan,_,_,_,calls,_,_=fixture(tmp_path);original=m.run_stage
    def crash(*a,**kw):original(*a,**kw);raise KeyboardInterrupt('explicit death after retained terminal output')
    monkeypatch.setattr(m,'run_stage',crash)
    with pytest.raises(KeyboardInterrupt):run()
    selected=(tmp_path/'sustained/derived/first/selection.json').read_bytes()
    before=len(calls);monkeypatch.setattr(m,'run_stage',original)
    assert run()['outcome']=='EXITED_ZERO' and len(calls)==before
    assert (tmp_path/'sustained/derived/first/selection.json').read_bytes()==selected


def test_final_result_is_adopted_after_crash_even_if_controller_then_stops(tmp_path,monkeypatch):
    run,_,_,_,_,calls,controller,_=fixture(tmp_path);original=m.finalize
    def crash(*a,**kw):raise KeyboardInterrupt('explicit death after final result before health completion')
    monkeypatch.setattr(m,'finalize',crash)
    with pytest.raises(KeyboardInterrupt):run()
    saved=read_json(tmp_path/'sustained/final/result.json');write_json(controller/'stop-request.json',{'later':'stop'})
    before=len(calls);monkeypatch.setattr(m,'finalize',original)
    assert run()==saved and len(calls)==before and read_json(tmp_path/'health.json')['complete']


@pytest.mark.parametrize('damage',['production','changed-template','overlap','unknown-parent','excess-time','stale-watchdog','stopped','wrong-profile'])
def test_invalid_plan_or_guard_refuses_without_remote_work(tmp_path,damage):
    run,plan,_,t,_,calls,controller,heartbeat=fixture(tmp_path,second=True)
    if damage=='production':
        p=tmp_path/plan['stages'][0]['template_path'];v=read_json(p);v['kind']='production-record';write_json(p,v);plan['stages'][0]['template_sha256']=digest(v)
    elif damage=='changed-template':plan['stages'][0]['template_sha256']='f'*64
    elif damage=='overlap':
        p=tmp_path/plan['stages'][1]['template_path'];v=read_json(p);v['export_roots']=[t.profile['remote_root']+'/output'];write_json(p,v);plan['stages'][1]['template_sha256']=digest(v)
    elif damage=='unknown-parent':plan['stages'][1]['parent_stage']='missing'
    elif damage=='excess-time':plan['stages'][0]['work_seconds']=1500
    elif damage=='stale-watchdog':
        v=read_json(heartbeat);v['observed_epoch']-=31;write_json(heartbeat,v)
    elif damage=='stopped':
        with Journal(controller).lease() as j:j.append('decision',{'action':'CHECKPOINT_AND_STOP'})
    else:t.profile['host']='127.0.0.2'
    with pytest.raises(EvidenceError):run()
    assert not calls


def test_failure_retains_all_outputs_and_never_launches_successor(tmp_path):
    run,plan,_,_,_,calls,_,_=fixture(tmp_path,second=True)
    p=tmp_path/plan['stages'][0]['template_path'];v=read_json(p);program=Path(v['argv'][1]);program.write_text('raise SystemExit(17)\n')
    v['required_files'][-1].update(bytes=program.stat().st_size,sha256=file_hash(program));write_json(p,v);plan['stages'][0]['template_sha256']=digest(v)
    result=run();assert result['outcome']=='FAILED_OR_ABANDONED' and result['unstarted_stages']==['second']
    assert result['stages'][0]['exit']['exit_code']==17 and read_json(tmp_path/'health.json')['complete']
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_journaled_stop_without_file_prevents_new_stage(tmp_path,monkeypatch):
    run,_,_,_,_,calls,controller,_=fixture(tmp_path,second=True);original=m.run_stage
    def stop_after(*a,**kw):
        result=original(*a,**kw)
        with Journal(controller).lease() as j:j.append('decision',{'action':'CHECKPOINT_AND_STOP'})
        return result
    monkeypatch.setattr(m,'run_stage',stop_after);result=run()
    assert result['outcome']=='STOPPED_AFTER_STAGE' and result['unstarted_stages']==['second']
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_altered_retained_terminal_bytes_cannot_complete_again(tmp_path):
    run,_,_,_,_,calls,_,_=fixture(tmp_path);run()
    (tmp_path/'sustained/stages/first/export-000/files/stdout.log').write_bytes(b'changed')
    before=len(calls)
    with pytest.raises(EvidenceError):run()
    assert len(calls)==before


@pytest.mark.parametrize('damage',['omitted-root','foreign-root','duplicate-root','extra-file','terminal','descriptor'])
def test_retention_is_bound_to_declared_roots_and_complete_terminal_bytes(tmp_path,damage):
    run,plan,_,t,_,calls,_,_=fixture(tmp_path);run();before=len(calls)
    stage=plan['stages'][0];path=tmp_path/'sustained/stages/first/stage-result.json';v=read_json(path)
    if damage=='omitted-root':v['exports'].pop()
    elif damage=='foreign-root':v['exports'][-1]['remote_root']='foreign'
    elif damage=='duplicate-root':v['exports'][-1]['remote_root']=v['exports'][0]['remote_root']
    elif damage=='extra-file':(Path(v['exports'][-1]['directory'])/'unlisted').write_bytes(b'unlisted evidence')
    elif damage=='terminal':v['exit']['exit_code']=17
    else:
        job=tmp_path/'sustained/derived/first/job.json';d=read_json(job);d['export_roots']=[];write_json(job,d)
    write_json(path,v)
    with pytest.raises(EvidenceError):m.retained_stage(stage,tmp_path/'sustained',t.profile)
    assert len(calls)==before


def test_bad_activity_stops_owned_process_retains_failure_and_never_relaunches(tmp_path):
    run,plan,_,t,remote,calls,_,_=fixture(tmp_path,second=True)
    p=tmp_path/plan['stages'][0]['template_path'];v=read_json(p);program=Path(v['argv'][1])
    program.write_text('from pathlib import Path\nimport time\nPath('+repr(str(remote/'output/activity.json'))+').write_text(\'{"bad":true}\')\nprint("retained before malformed telemetry",flush=True)\ntime.sleep(20)\n')
    v['required_files'][-1].update(bytes=program.stat().st_size,sha256=file_hash(program))
    v['environment']['OVL_ACTIVITY_FILE']=t.profile['remote_root']+'/output/activity.json'
    write_json(p,v);plan['stages'][0]['template_sha256']=digest(v)
    result=run()
    assert result['outcome']=='DISPATCH_FAILED_AFTER_RETENTION' and result['unstarted_stages']==['second']
    assert result['failure']['error_type']=='EvidenceError' and result['stages'][0]['exit']['exit_code']<0
    diagnostics=list((tmp_path/'sustained/stages/first/private-transport-diagnostics').glob('*.json'))
    assert len(diagnostics)==1 and read_json(diagnostics[0])['context']['phase']=='stage-dispatch'
    assert all('private-transport-diagnostics' not in item['path'] for export in read_json(tmp_path/'sustained/stages/first/stage-result.json')['exports'] for item in export['files'])
    assert (tmp_path/'sustained/stages/first/export-001/files/activity.json').read_bytes()==b'{"bad":true}'
    assert read_json(tmp_path/'health.json')['complete'] is True
    before=len(calls);assert run()==result and len(calls)==before
    assert len([c for c in calls if ' start ' in c[-1]])==1


def test_cpu_substituted_record_full_replay_and_all_historical_binding_adoption(cpu_runtime,prepared,tmp_path,monkeypatch):
    """Real numerical CPU fixture and SSH byte copies; explicit exit/process doubles."""
    from ovl_pipeline import gpu_pilot,runtime_activity,training
    from ovl_pipeline.fixture import recipe
    from ovl_pipeline.canonical import inventory
    from pod_versioned_export import export
    from sustained_pilot_selection import RECORD
    run,plan,_,t,remote,calls,_,_=fixture(tmp_path,second=True)
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size']);kernel={'schema':'ovl.gpu-kernel.v1','precision':'fp32'}
    binding={'schema':'ovl.pilot-record-parent-binding.v1','recipe_sha256':digest(r),'kernel_sha256':digest(kernel),
             'stream_sha256':digest(manifest['streams']['wikipedia']),'code_root':training.code_root()}
    for i,s in enumerate(plan['stages']):
        name=s['name'];mode='record' if i==0 else 'replay';p=tmp_path/s['template_path'];v=read_json(p)
        root=t.profile['remote_root']+'/'+mode;control=t.profile['remote_root']+'/control-'+mode
        v['kind']='pilot';v['export_roots']=[root,control];v['environment']['OVL_ACTIVITY_FILE']=control+'/activity.json'
        v['argv']+=[mode]
        if i:v['argv']+=['--expected-record-sha256',RECORD]
        write_json(p,v);s.update(template_sha256=digest(v),parent_binding=dict(binding),
            validation_binding={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':binding['stream_sha256'],
                                'documents':manifest['streams']['wikipedia']['documents']},
            retention={'schema':'ovl.pilot-initial-retention.v1','mode':mode,'output_root':root,'phase':'wikipedia','maximum_initial_bytes':1024**2})
        if i:s.update(parent_stage='first',parent_record_root=t.profile['remote_root']+'/record')
    executions=[]
    def numerical_stage(transport,h,job_file,job_root,worker,worker_root,out,health_file,stop,rental_root,*,sleep,initial_retention):
        job=read_json(job_file);mode=initial_retention.selection['mode'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':job_root,'pod_id':h.pod,'kind':'pilot'})
        if (out/'stage-result.json').exists():return m.saved_result(out/'stage-result.json',job_root)
        executions.append(mode)
        # Check derived record-required bytes, as the actual worker does; no
        # unselected or missing remote parent can pass the synthetic job here.
        for f in job['required_files']:
            path=remote/f['path'][len(t.profile['remote_root'])+1:] if f['path'].startswith(t.profile['remote_root']+'/') else Path(f['path'])
            assert path.stat().st_size==f['bytes'] and file_hash(path)==f['sha256']
        control=remote/('control-'+mode);control.mkdir()
        monkeypatch.setenv('OVL_ACTIVITY_FILE',str(control/'activity.json'))
        for key,value in (('_last',None),('_sequence',0),('_process',None)):monkeypatch.setattr(runtime_activity,key,value)
        if mode=='record':result=gpu_pilot.record(directory/'wikipedia',r,kernel,remote/'record',updates=6,checkpoint_every=3)
        else:result=gpu_pilot.replay(directory/'wikipedia',remote/'record',job['argv'][-1],remote/'replay')
        h.activity(job_root,read_json(control/'activity.json'))
        initial=remote/m.remote_name(t,initial_retention.selection['output_root'])/('boundary-00000' if mode=='record' else 'verifier-boundary-00000')
        initial_retention.observe({initial_retention.marker:read_json(initial/'checkpoint.json')})
        status={'schema':'ovl.workload-job-exit.v1','job_sha256':job_root,'state':'EXITED','exit_code':0}
        metadata=remote/'jobs'/job_root;metadata.mkdir(parents=True);write_json(metadata/'exit.json',status)
        exports=[]
        for index,name in enumerate(['jobs/'+job_root,*[m.remote_name(t,p) for p in job['export_roots']]]):
            receipt=export(t,name,initial_retention.store,out/f'export-{index:03d}',int(time.time())+30)
            h.exported_files(job_root,Path(receipt['files_directory']),receipt['files'])
            exports.append({'remote_root':name,'directory':receipt['files_directory'],'files':receipt['files']})
        retained={'schema':'ovl.workload-stage-result.v1','job_sha256':job_root,'exit':status,'exports':exports,
                  'scope':'explicit synthetic CPU orchestration and exit double; no real child/CUDA acceptance'}
        write_json(out/'stage-result.json',retained);h.job_exit(job_root,status);h.write(health_file);return retained
    monkeypatch.setattr(m,'run_stage',numerical_stage)
    result=run();assert result['outcome']=='EXITED_ZERO' and executions==['record','replay']
    replay=read_json(remote/'replay/verification.json');assert replay['updates_recomputed']==6 and len(replay['compared'])==3
    before=len(calls);assert run()==result and len(calls)==before and executions==['record','replay']
