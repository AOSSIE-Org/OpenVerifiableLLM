"""Initializer orchestration with real CPU states/transfers and explicit process doubles."""
from pathlib import Path
import sys
import time
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_sustained_pilot as m
from test_sustained_pilot_dispatch import fixture
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from test_initialization import distinct_process_observations
from ovl_pipeline import initialization,training,runtime_activity,production_observation
from ovl_pipeline.fixture import recipe
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json,file_hash
from pod_versioned_export import export
from sustained_pilot_selection import RECORD


def selected(tmp_path,prepared):
    run,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True)
    data,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size']);kernel={'schema':'ovl.gpu-kernel.v1','precision':'bf16'}
    # External parameter expectation from the actual selected architecture.
    from model import TinyGPT
    model=TinyGPT(**r['model']);model.lm_head.weight=model.transformer['wte'].weight
    count=sum(p.numel() for p in model.parameters())
    binding={'schema':'ovl.initialization-record-parent-binding.v1','recipe_sha256':digest(r),
        'kernel_sha256':digest(kernel),'stream_sha256':digest(manifest['streams']['wikipedia']),
        'code_root':training.code_root(),'parameter_count':count,'warmup_updates':2}
    plan['schema']='ovl.initialization-cycle-plan.v1'
    for i,s in enumerate(plan['stages']):
        action='record' if i==0 else 'verify';path=tmp_path/s['template_path'];job=read_json(path)
        root=t.profile['remote_root']+'/'+action;control=t.profile['remote_root']+'/control-'+action
        job['kind']='pilot';job['export_roots']=[root,control];job['environment']['OVL_ACTIVITY_FILE']=control+'/activity.json'
        job['argv']+=['--',action,'--output',root]
        if i:job['argv']+=['--expected-record-sha256',RECORD]
        write_json(path,job)
        s.update(template_sha256=digest(job),parent_binding=dict(binding),
            validation_binding={'schema':'ovl.initialization-validation-binding.v1','stream_sha256':binding['stream_sha256'],
                'documents':manifest['streams']['wikipedia']['documents'],'action':action})
        if i:s.update(parent_stage='first',parent_record_root=t.profile['remote_root']+'/record')
    return run,plan,rental,t,remote,calls,r,kernel,data


@pytest.mark.parametrize('interrupted',[False,True])
@pytest.mark.parametrize('phase_only',[False,True])
def test_actual_cpu_initial_states_and_full_retention_precede_regeneration_and_adoption(
        tmp_path,prepared,cpu_runtime,distinct_process_observations,monkeypatch,interrupted,phase_only):
    run,plan,rental,t,remote,calls,r,kernel,data=selected(tmp_path,prepared);executions=[]
    if phase_only:
        from production_run_health import ProductionRunHealth
        from ovl_pipeline.supervision import Journal
        plan.update(schema='ovl.initialization-cycle-plan.v2',prior_jobs=[],
                    completion_scope='phase-only-under-production-run-coordinator')
        def run():
            bindings={}
            # Recover every selected contract before adopting the shared journal.
            for selected_stage in plan['stages']:
                p=tmp_path/'sustained/derived'/selected_stage['name']/'selection.json'
                if p.exists():bindings[read_json(p)['job_sha256']]=selected_stage['validation_binding']
            with Journal(tmp_path/'run-health').lease() as journal:
                h=ProductionRunHealth(journal,rental['watchdog_intent'],t.profile['pod_id'],None,bindings,{})
                result=m.run(plan,digest(plan),rental,tmp_path/'controller',tmp_path/'watchdog.json',t,tmp_path,
                    Path(__file__).parents[1]/'scripts/pod_job_worker.py',tmp_path/'sustained',tmp_path/'health.json',
                    run_health=h,sleep=lambda _:time.sleep(.01))
                assert not h.complete and not read_json(tmp_path/'health.json')['complete']
                assert read_json(tmp_path/'sustained/phase-retention.json')['rental_complete'] is False
                return result
    def stage(transport,h,job_file,root,worker,worker_root,out,health_file,stop,rental_root,*,sleep,initial_retention):
        assert initial_retention is None
        job=read_json(job_file);args=job['argv'][job['argv'].index('--')+1:];action=args[0]
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        if (out/'stage-result.json').exists():return m.saved_result(out/'stage-result.json',root)
        executions.append(action)
        for f in job['required_files']:
            path=remote/f['path'][len(t.profile['remote_root'])+1:] if f['path'].startswith(t.profile['remote_root']+'/') else Path(f['path'])
            assert path.stat().st_size==f['bytes'] and file_hash(path)==f['sha256']
        control=remote/('control-'+action);control.mkdir()
        monkeypatch.setenv('OVL_ACTIVITY_FILE',str(control/'activity.json'))
        for key,value in (('_last',None),('_sequence',0),('_process',None)):monkeypatch.setattr(runtime_activity,key,value)
        monkeypatch.setattr(production_observation,'_pass_index',0)
        if action=='record':initialization.record(data/'wikipedia',r,kernel,remote/'record',warmup_updates=2)
        else:initialization.verify(data/'wikipedia',remote/'record',args[args.index('--expected-record-sha256')+1],remote/'verify')
        h.activity(root,read_json(control/'activity.json'))
        status={'schema':'ovl.workload-job-exit.v1','job_sha256':root,'state':'EXITED','exit_code':0}
        meta=remote/'jobs'/root;meta.mkdir(parents=True);write_json(meta/'exit.json',status)
        exports=[]
        for index,name in enumerate(['jobs/'+root,*[m.remote_name(t,p) for p in job['export_roots']]]):
            receipt=export(t,name,tmp_path/'objects',out/f'export-{index:03d}',int(time.time())+30)
            h.exported_files(root,Path(receipt['files_directory']),receipt['files'])
            exports.append({'remote_root':name,'directory':receipt['files_directory'],'files':receipt['files']})
        result={'schema':'ovl.workload-stage-result.v1','job_sha256':root,'exit':status,'exports':exports,
                'scope':'actual CPU states and local byte transfers; explicit child/process/CUDA substitutes'}
        write_json(out/'stage-result.json',result);h.job_exit(root,status);h.write(health_file);return result
    monkeypatch.setattr(m,'run_stage',stage)
    if interrupted:
        import copy
        original_finalize=m.finalize
        def crash(*args,**kwargs):raise KeyboardInterrupt('explicit coordinator interruption before final health commit')
        monkeypatch.setattr(m,'finalize',crash)
        with pytest.raises(KeyboardInterrupt):run()
        monkeypatch.setattr(m,'finalize',original_finalize)
        stage_path=tmp_path/'sustained/stages/second/stage-result.json'
        final_path=tmp_path/'sustained/final/result.json'
        original_stage=read_json(stage_path);original_final=read_json(final_path)
        changed=copy.deepcopy(original_stage);changed['exports'][-1]['remote_root']='foreign-audit'
        changed_final=copy.deepcopy(original_final);changed_final['stages'][-1]['result_sha256']=digest(changed)
        write_json(stage_path,changed);write_json(final_path,changed_final)
        with pytest.raises(EvidenceError):run()
        write_json(stage_path,original_stage);write_json(final_path,original_final)
    result=run();assert result['outcome']=='EXITED_ZERO' and executions==['record','verify']
    assert read_json(tmp_path/'sustained/initialization-consistency.json')['result']=='PASS'
    before=len(calls);assert run()==result and len(calls)==before and executions==['record','verify']
    # A locally rewritten report must not relabel a retained audit as another
    # root even if its bytes and the updated enclosing report hash still match.
    import copy
    stage_path=tmp_path/'sustained/stages/second/stage-result.json'
    final_path=tmp_path/'sustained/final/result.json'
    original_stage=read_json(stage_path);original_final=read_json(final_path)
    changed=copy.deepcopy(original_stage);changed['exports'][-1]['remote_root']='foreign-audit'
    changed_final=copy.deepcopy(original_final);changed_final['stages'][-1]['result_sha256']=digest(changed)
    write_json(stage_path,changed);write_json(final_path,changed_final)
    with pytest.raises(EvidenceError):run()
    write_json(stage_path,original_stage);write_json(final_path,original_final)
    retained=read_json(tmp_path/'sustained/stages/first/stage-result.json')
    directory=Path(next(e['directory'] for e in retained['exports'] if e['remote_root']=='record'))
    state=directory/'initial-state/state.safetensors';state.chmod(0o600);state.write_bytes(b'altered after completion')
    with pytest.raises(EvidenceError):run()
    assert len(calls)==before


@pytest.mark.parametrize('damage',['action','binding','mode','missing_verify','root','retention','export_age','production'])
def test_mixed_or_unbounded_initialization_plan_refused_before_launch(tmp_path,prepared,damage):
    run,plan,_,t,_,calls,_,_,_=selected(tmp_path,prepared)
    s=plan['stages'][1]
    if damage=='action':s['validation_binding']['action']='record'
    elif damage=='binding':s['parent_binding']['recipe_sha256']='e'*64
    elif damage=='mode':s['validation_binding']['schema']='ovl.pilot-validation-binding.v1'
    elif damage=='missing_verify':plan['stages']=plan['stages'][:1]
    elif damage=='root':s['parent_record_root']=t.profile['remote_root']+'/control-record'
    elif damage=='retention':s['retention']={'schema':'unselected'}
    elif damage=='export_age':s.update(work_seconds=1500,export_reserve_seconds=600)
    else:
        p=tmp_path/s['template_path'];job=read_json(p);job['kind']='production-record';write_json(p,job);s['template_sha256']=digest(job)
    with pytest.raises((EvidenceError,KeyError)):run()
    assert not calls
