"""Real safe-state transfers and CPU-substituted numerical replay, never CUDA credit."""
import copy
from pathlib import Path
import time
import pytest

from test_gpu_pilot import cpu_runtime
from test_pipeline import prepared
from test_pod_transfer import setup
from test_workload_stage import intent
from ovl_pipeline import gpu_pilot, pilot_delivery as d, training
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.fixture import recipe
from ovl_pipeline.state import save_state
from ovl_pipeline.supervision import Journal
from workload_health import Health
from pilot_checkpoint_retention import CheckpointRetention,selected_policy
from pilot_record_parent import check as check_record
from verify_pilot_cycle import replay_states


def selected(t,mode='record',binding=None):
    return {'schema':'ovl.pilot-checkpoint-retention.v1','session':('a' if mode=='record' else 'b')*64,
            'mode':mode,'output_root':t.profile['remote_root']+'/'+mode,'phase':'wikipedia',
            'maximum_checkpoint_bytes':10*1024**2,'copy_timeout_seconds':120,'maximum_uncached_export_bytes':1024**2,
            'binding':binding or {'schema':'ovl.pilot-record-parent-binding.v1','recipe_sha256':'1'*64,
              'kernel_sha256':'2'*64,'stream_sha256':'3'*64,'code_root':'4'*64}}


def job_for(t,s,record=None):
    deadline=int(time.time())+240
    argv=['test','--',s['mode'],'--output',s['output_root'],'--delivery-session',s['session'],
          '--delivery-deadline',str(deadline),'--delivery-timeout',str(s['copy_timeout_seconds']),
          '--delivery-maximum-bytes',str(s['maximum_checkpoint_bytes'])]
    if record is not None:argv+=['--expected-record-sha256',record]
    return {'kind':'pilot','export_roots':[s['output_root']],'deadline_epoch':deadline,'argv':argv}


def request_for(remote,s,job,*,index=0,previous=None):
    m,o,c=training.initialize(recipe(300));c.update(phase=s['phase'],pilot_cycle=0,global_step=index,phase_step=index)
    path=f'boundary-{index:05d}';cp=save_state(remote/'record'/path,m,o,c)
    p=selected_policy(s,job['deadline_epoch'])
    return {'schema':'ovl.pilot-delivery-request.v1','policy':p,'origin_sha256':d.origin(s['binding']),
            'index':index,'previous':previous or digest(p),'path':path,'checkpoint':cp,'control':c,
            'copy_deadline_epoch':int(time.time())+s['copy_timeout_seconds']}


def hook_for(t,h,job,s,tmp):
    root=digest(job)
    if root not in h.jobs:h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
    return CheckpointRetention(t,h,job,root,s,tmp/'retention',tmp/'health.json',tmp/'store')


@pytest.mark.parametrize('profile_timing',[False,True])
def test_actual_record_full_replay_with_delivery_and_fresh_safe_state_checks(cpu_runtime,prepared,tmp_path,monkeypatch,profile_timing):
    t,remote,calls,_=setup(tmp_path);directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size'])
    kernel={'schema':'ovl.gpu-kernel.v1','precision':'fp32'}
    binding=d.binding({'recipe':r,'kernel':kernel,'stream':read_json(directory/'wikipedia/stream.json'),'code_root':gpu_pilot.code_root()})
    original=d.Delivery;reports={};all_files={}
    for mode in ('record','replay'):
        area=tmp_path/mode;area.mkdir();s=selected(t,mode,binding);job=job_for(t,s,None if mode=='record' else digest(reports['record']))
        if profile_timing:job['argv'].append('--profile-timing')
        with Journal(area/'journal').lease() as journal:
            h=Health(journal,intent(),t.profile['pod_id']);hook=hook_for(t,h,job,s,area)
            def sender(output,policy,origin):
                def copy_ack(_):
                    time.sleep(.04) # Explicit local transport delay, included in pilot timings.
                    hook.observe({hook.marker:read_json(output/'delivery/request.json')})
                return original(output,policy,origin,sleep=copy_ack)
            monkeypatch.setattr(d,'Delivery',sender)
            if mode=='record':
                result=gpu_pilot.record(directory/'wikipedia',r,kernel,remote/mode,updates=5,checkpoint_every=2,delivery=selected_policy(s,job['deadline_epoch']))
            else:
                with monkeypatch.context() as m:
                    m.setattr(gpu_pilot,'restore',lambda *a:pytest.fail('full replay may not restore prover state'))
                    result=gpu_pilot.replay(directory/'wikipedia',remote/'record',digest(reports['record']),remote/mode,delivery=selected_policy(s,job['deadline_epoch']))
            assert len(h.exports)==4 and hook.index==4 and result['measured_ms']>=120
            if profile_timing:
                from ovl_pipeline.phase_timing import validate
                for index in range(4):
                    checkpoint=area/'retention'/f'checkpoint-{index:05d}'
                    profile=read_json(checkpoint/'timing.json')
                    validate(profile,{'job_sha256':digest(job),'request_sha256':digest(read_json(checkpoint/'request.json'))},
                        scope='operator-checkpoint-controller-completed-attempt-only')
                    names={e['operation'] for e in profile['measurements']}
                    assert {'safe_state_verification_and_health','acknowledgement_transfer','export_inventory_hashing_and_storage'}<=names
            before=len(calls);credited=h.exported
            adopted=hook_for(t,h,job,s,area)
            assert not adopted.observe({hook.marker:read_json(remote/mode/'delivery/request.json')})
            assert len(calls)==before and h.exported==credited
        reports[mode]=result
        all_files[mode]=inventory(remote/mode,[p.relative_to(remote/mode).as_posix() for p in (remote/mode).rglob('*') if p.is_file()])
    assert reports['replay']['updates_recomputed']==5
    assert check_record(remote/'record',all_files['record'],binding)['record_sha256']==digest(reports['record'])
    assert replay_states(remote/'replay',all_files['replay'],reports['record'],digest(reports['record']),resume_from=None)['actual_safe_states']==4
    resume=gpu_pilot.replay(directory/'wikipedia',remote/'record',digest(reports['record']),remote/'resume',resume_from=1)
    resume_files=inventory(remote/'resume',[p.relative_to(remote/'resume').as_posix() for p in (remote/'resume').rglob('*') if p.is_file()])
    assert resume['updates_recomputed']==3 and resume['eligible_for_forecast_comparison'] is False
    assert replay_states(remote/'resume',resume_files,reports['record'],digest(reports['record']),resume_from=1)['actual_safe_states']==4
    with pytest.raises(EvidenceError,match='requires replay delivery'):
        gpu_pilot.replay(directory/'wikipedia',remote/'record',digest(reports['record']),remote/'missing-delivery')
    bad=read_json(remote/'replay/delivery/ack-00001.json');bad['state_root']='0'*64
    write_json(remote/'replay/delivery/ack-00001.json',bad)
    all_files['replay']=inventory(remote/'replay',[p.relative_to(remote/'replay').as_posix() for p in (remote/'replay').rglob('*') if p.is_file()])
    with pytest.raises(EvidenceError,match='acknowledgement'):
        replay_states(remote/'replay',all_files['replay'],reports['record'],digest(reports['record']),resume_from=None)


@pytest.mark.parametrize('damage',['session','origin','index','previous','phase','root','control','extra','partial','budget','expired'])
def test_changed_requests_or_bad_bytes_never_acknowledged(tmp_path,damage):
    t,remote,calls,_=setup(tmp_path);s=selected(t);job=job_for(t,s);q=request_for(remote,s,job)
    if damage=='session':q['policy']['session']='c'*64
    elif damage=='origin':q['origin_sha256']='c'*64
    elif damage=='index':q['index']=1
    elif damage=='previous':q['previous']='c'*64
    elif damage=='phase':q['control']['phase']='conversation'
    elif damage=='root':q['checkpoint']['state_root']='c'*64
    elif damage=='control':q['control']['transcript']='c'*64
    elif damage=='extra':(remote/'record/boundary-00000/extra').write_bytes(b'not selected')
    elif damage=='partial':(remote/'record/boundary-00000/state.safetensors').write_bytes(b'partial')
    elif damage=='budget':s['maximum_checkpoint_bytes']=1;job=job_for(t,s);q['policy']=selected_policy(s,job['deadline_epoch']);q['previous']=digest(q['policy'])
    else:q['copy_deadline_epoch']=int(time.time())-1
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);hook=hook_for(t,h,job,s,tmp_path)
        with pytest.raises(EvidenceError):hook.observe({hook.marker:q})
        assert not h.exports and not (remote/'record/delivery/ack-00000.json').exists()


def test_interrupted_transfer_retains_bytes_and_cannot_renew_deadline(tmp_path):
    from pod_transfer import TransientTransportError
    t,remote,calls,_=setup(tmp_path);s=selected(t);job=job_for(t,s);q=request_for(remote,s,job);get=t.get
    def broken(name,dest,*a,**k):
        dest.with_name(dest.name+'.partial').write_bytes(b'')
        error=TransientTransportError('interrupted');error.transfer_counts={'bytes_sent':0,'bytes_received':0};raise error
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);hook=hook_for(t,h,job,s,tmp_path);t.get=broken
        with pytest.raises(EvidenceError,match='interrupted'):hook.observe({hook.marker:q})
        changed=copy.deepcopy(q);changed['copy_deadline_epoch']-=1
        with pytest.raises(EvidenceError):hook.observe({hook.marker:changed})
        t.get=get;hook=hook_for(t,h,job,s,tmp_path);assert hook.observe({hook.marker:q})
        assert list((tmp_path/'store/incoming').glob('*/verified.partial'))
        assert (tmp_path/'retention/checkpoint-00000/snapshot-001/export.json').exists()
        before=len(calls);age=h.exported
        assert not hook.observe({hook.marker:q}) and len(calls)==before and h.exported==age
        with pytest.raises(EvidenceError):hook.observe({hook.marker:None})


def test_lost_ack_response_is_adopted_without_recopy_or_second_export_credit(tmp_path):
    t,remote,calls,_=setup(tmp_path);s=selected(t);job=job_for(t,s);q=request_for(remote,s,job);put=t.put
    def lost(*a,**k):put(*a,**k);raise EvidenceError('lost response')
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);hook=hook_for(t,h,job,s,tmp_path);t.put=lost
        with pytest.raises(EvidenceError,match='lost response'):hook.observe({hook.marker:q})
        assert len(h.exports)==1 and (remote/'record/delivery/ack-00000.json').exists()
        t.put=put;hook=hook_for(t,h,job,s,tmp_path);before=len(calls);age=h.exported
        assert not hook.observe({hook.marker:q}) and len(calls)==before+1 and h.exported==age


@pytest.mark.parametrize('failure',['authentication','identity','integrity','unknown','range-exhausted','missing-classification'])
def test_restart_cannot_reclassify_a_fatal_or_unknown_checkpoint_failure(tmp_path,failure):
    from pod_transfer import RangeRecoveryExhausted
    t,remote,calls,_=setup(tmp_path);s=selected(t);job=job_for(t,s);q=request_for(remote,s,job);original=t.get
    def broken(name,dest,*a,**k):
        dest.with_name(dest.name+'.partial').write_bytes(b'private synthetic failure evidence')
        if failure=='range-exhausted':raise RangeRecoveryExhausted('range retries exhausted')
        raise EvidenceError(failure)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);hook=hook_for(t,h,job,s,tmp_path);t.get=broken
        with pytest.raises(EvidenceError):hook.observe({hook.marker:q})
        if failure=='missing-classification':(tmp_path/'retention/checkpoint-00000/snapshot-000/failure.json').unlink()
        t.get=original;hook=hook_for(t,h,job,s,tmp_path);before=len(calls)
        with pytest.raises(EvidenceError,match='classification|fatal'):hook.observe({hook.marker:q})
        assert len(calls)==before and not h.exports
        assert not(tmp_path/'retention/checkpoint-00000/snapshot-001').exists()
        assert not(remote/'record/delivery/ack-00000.json').exists()


@pytest.mark.parametrize('damage',['session','request_sha256','index','state_root','scope'])
def test_worker_rejects_foreign_ack_before_next_update(tmp_path,damage):
    s={'schema':'ovl.pilot-delivery-policy.v1','session':'a'*64,'mode':'record','phase':'wikipedia',
       'deadline_epoch':200,'copy_timeout_seconds':30,'maximum_checkpoint_bytes':10*1024**2}
    out=tmp_path/'record';m,o,c=training.initialize(recipe(300));c.update(phase='wikipedia',pilot_cycle=0)
    cp=save_state(out/'boundary-00000',m,o,c)
    def ack(_):
        q=read_json(out/'delivery/request.json');a=d.acknowledgement(q,'c'*64)
        a[damage]=1 if damage=='index' else 'b'*64;write_json(out/'delivery/ack-00000.json',a)
    sender=d.Delivery(out,s,'b'*64,wall=lambda:100,sleep=ack)
    with pytest.raises(EvidenceError,match='acknowledgement'):sender.checkpoint('boundary-00000',cp,c)
    assert sender.index==0


def test_worker_times_out_without_renewing_request(tmp_path):
    s={'schema':'ovl.pilot-delivery-policy.v1','session':'a'*64,'mode':'record','phase':'wikipedia',
       'deadline_epoch':200,'copy_timeout_seconds':30,'maximum_checkpoint_bytes':10*1024**2}
    out=tmp_path/'record';m,o,c=training.initialize(recipe(300));c.update(phase='wikipedia',pilot_cycle=0)
    cp=save_state(out/'boundary-00000',m,o,c);clock=[100]
    sender=d.Delivery(out,s,'b'*64,wall=lambda:clock[0],sleep=lambda _:clock.__setitem__(0,clock[0]+10))
    with pytest.raises(EvidenceError,match='deadline expired'):sender.checkpoint('boundary-00000',cp,c)
    assert read_json(out/'delivery/request.json')['copy_deadline_epoch']==130 and sender.index==0


@pytest.mark.parametrize('work_seconds',[240,2700])
def test_detached_stage_waits_for_checked_ack_and_bounded_terminal_export(tmp_path,work_seconds):
    from test_workload_stage import staged
    from ovl_pipeline.canonical import file_hash
    import run_workload_stage as stages
    t,remote,calls,job_file,_,worker,worker_root=staged(tmp_path)
    s=selected(t);job=read_json(job_file);flags=job_for(t,s)
    flags['deadline_epoch']=int(time.time())+work_seconds
    flags['argv'][flags['argv'].index('--delivery-deadline')+1]=str(flags['deadline_epoch'])
    job.update(deadline_epoch=flags['deadline_epoch'],export_roots=flags['export_roots'])
    script=Path(job['argv'][1]);job['argv']+=flags['argv'][1:]
    q=request_for(remote,s,job)
    delivery=remote/'record/delivery';delivery.mkdir();write_json(delivery/'request.json',q)
    script.write_text('from pathlib import Path\nimport time\np=Path('+repr(str(delivery/'ack-00000.json'))+')\nwhile not p.exists():time.sleep(.01)\nprint("acknowledged checkpoint; complete terminal log",flush=True)\n')
    job['required_files'][-1].update(bytes=script.stat().st_size,sha256=file_hash(script));write_json(job_file,job)
    from datetime import datetime,timezone
    from ovl_pipeline.supervision import rental_plan
    w=intent();w['plan']=rental_plan({**w['plan']['input'],'maximum_seconds':7200})
    w['payload']['terminateAfter']=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,w,t.profile['pod_id']);hook=hook_for(t,h,job,s,tmp_path)
        kwargs={'sleep':lambda _:time.sleep(.02),'initial_retention':hook,
                'terminal_limits':{'maximum_bytes':20*1024**2,'maximum_uncached_bytes':1024**2}}
        result=stages.run_stage(t,h,job_file,digest(job),worker,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop','d'*64,**kwargs)
        assert result['exit']['exit_code']==0 and h.jobs[digest(job)]['finished']
        assert 'acknowledged checkpoint' in (Path(result['exports'][0]['directory'])/'stdout.log').read_text()
        receipt=read_json(tmp_path/'stage/export-001/export.json')
        assert receipt['schema']=='ovl.offpod-versioned-tree-export.v2'
        assert 'boundary-00000/state.safetensors' in receipt['reused_paths']
        before=len(calls)
        assert stages.run_stage(t,h,job_file,digest(job),worker,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop','d'*64,**kwargs)==result
        assert len(calls)==before


@pytest.mark.parametrize('damage',[None,'long-valid','command','binding','copy-reserve','uncached-bound','phase-duration'])
def test_delivery_plan_admission_preserves_timing_and_parent_requirements(tmp_path,damage):
    from test_sustained_pilot_dispatch import fixture
    from sustained_pilot_selection import DEADLINE
    import run_sustained_pilot as runner
    _,plan,rental,t,remote,calls,_,_=fixture(tmp_path)
    stage=plan['stages'][0];path=tmp_path/stage['template_path'];job=read_json(path)
    s=selected(t);s['output_root']=job['export_roots'][0]
    stage.update(retention=s,work_seconds=60,maximum_export_bytes=20*1024**2,
                 parent_binding=s['binding'],validation_binding={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':'3'*64,'documents':1})
    args=job_for(t,s)['argv'][1:];args[args.index('--delivery-deadline')+1]=DEADLINE
    job['argv']+=args;job['kind']='pilot';job['environment']['OVL_ACTIVITY_FILE']=s['output_root']+'/activity.json'
    if damage=='command':job['argv'][job['argv'].index('--delivery-session')+1]='c'*64
    elif damage=='binding':stage['parent_binding']={**s['binding'],'recipe_sha256':'d'*64}
    elif damage=='copy-reserve':s['copy_timeout_seconds']=30;job['argv'][job['argv'].index('--delivery-timeout')+1]='30'
    elif damage=='uncached-bound':s['maximum_uncached_export_bytes']=stage['maximum_export_bytes']+1
    elif damage in ('long-valid','phase-duration'):
        from datetime import datetime,timezone
        from ovl_pipeline.supervision import rental_plan
        w=rental['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'maximum_seconds':7200})
        stamp=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        w['payload']['terminateAfter']=rental['payload']['terminateAfter']=stamp
        plan['rental_intent_sha256']=digest(rental)
        stage['work_seconds']=2700 if damage=='long-valid' else 2701
    write_json(path,job);stage['template_sha256']=digest(job)
    worker=Path(__file__).parents[1]/'scripts/pod_job_worker.py'
    if damage in (None,'long-valid'):assert runner.validate(plan,digest(plan),rental,t,tmp_path,worker)[2][0][0]==stage
    else:
        with pytest.raises(EvidenceError):runner.validate(plan,digest(plan),rental,t,tmp_path,worker)
    assert not calls
