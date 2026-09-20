from pathlib import Path
import time
import pytest

from test_pod_transfer import setup
from test_workload_stage import intent,staged
from ovl_pipeline import training
from ovl_pipeline.fixture import recipe
from ovl_pipeline.state import save_state
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal
from workload_health import Health
from pilot_retention import InitialRetention


def state(path,*,step=0):
    model,opt,control=training.initialize(recipe(300))
    control.update(phase='wikipedia',pilot_cycle=0,global_step=step)
    return save_state(path,model,opt,control)


def selected(t):
    return {'schema':'ovl.pilot-initial-retention.v1','mode':'record','output_root':t.profile['remote_root']+'/record',
            'phase':'wikipedia','maximum_initial_bytes':10*1024**2}


def job_for(t):return {'kind':'pilot','export_roots':[t.profile['remote_root']+'/record']}


def test_actual_safe_state_export_is_credited_once_and_adopted_without_retransfer(tmp_path):
    t,remote,calls,_=setup(tmp_path);marker=state(remote/'record/boundary-00000');job=job_for(t);root=digest(job)
    w=intent();output=tmp_path/'initial';health_file=tmp_path/'health.json'
    with Journal(tmp_path/'j').lease() as j:
        h=Health(j,w,t.profile['pod_id']);h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        hook=InitialRetention(t,h,job,root,selected(t),output,health_file,tmp_path/'store')
        assert hook.observe({hook.marker:marker});before=len(calls);credited=h.exported
        assert not hook.observe({hook.marker:marker}) and len(calls)==before
        assert not h.complete and len(h.exports)==1
    with Journal(tmp_path/'j').lease() as j:
        h=Health(j,w,t.profile['pod_id']);hook=InitialRetention(t,h,job,root,selected(t),output,health_file,tmp_path/'store')
        assert not hook.observe({hook.marker:marker}) and len(calls)==before and h.exported==credited
        with pytest.raises(EvidenceError):hook.observe({hook.marker:None})
    assert (output/'snapshot-000/files/state.safetensors').is_file()


@pytest.mark.parametrize('damage',['altered','extra','advanced','budget','foreign-marker'])
def test_bad_or_incomplete_safe_states_get_no_export_credit(tmp_path,damage):
    t,remote,calls,_=setup(tmp_path);path=remote/'record/boundary-00000'
    marker=state(path,step=int(damage=='advanced'));job=job_for(t);root=digest(job);selection=selected(t)
    if damage=='altered':(path/'state.safetensors').write_bytes(b'corrupt')
    elif damage=='extra':(path/'unexpected').write_bytes(b'omitted from state')
    elif damage=='budget':selection['maximum_initial_bytes']=1
    elif damage=='foreign-marker':marker={**marker,'state_root':'f'*64}
    with Journal(tmp_path/'j').lease() as j:
        h=Health(j,intent(),t.profile['pod_id']);h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        hook=InitialRetention(t,h,job,root,selection,tmp_path/'initial',tmp_path/'health.json',tmp_path/'store')
        with pytest.raises(EvidenceError):hook.observe({hook.marker:marker})
        assert not h.exports and not (tmp_path/'initial/retained.json').exists()
        if damage=='advanced':assert (tmp_path/'initial/snapshot-000/files').exists()
        else:assert not list((tmp_path/'store').glob('incoming/*/verified'))


def test_interrupted_copy_keeps_partial_bytes_and_bounded_fresh_retry(tmp_path):
    t,remote,calls,_=setup(tmp_path);marker=state(remote/'record/boundary-00000');job=job_for(t);root=digest(job)
    original=t.get
    def fail(name,dest,*a,**kw):
        dest.with_name(dest.name+'.partial').write_bytes(b'preserved interrupted evidence')
        raise EvidenceError('injected interruption')
    with Journal(tmp_path/'j').lease() as j:
        h=Health(j,intent(),t.profile['pod_id']);h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'pilot'})
        hook=InitialRetention(t,h,job,root,selected(t),tmp_path/'initial',tmp_path/'health.json',tmp_path/'store')
        t.get=fail
        with pytest.raises(EvidenceError):hook.observe({hook.marker:marker})
        assert not h.exports
        t.get=original;assert hook.observe({hook.marker:marker})
        assert list((tmp_path/'store/incoming').glob('*/verified.partial'))
        assert (tmp_path/'initial/snapshot-001/export.json').exists()


def test_terminal_whole_output_export_reuses_checked_state_and_preserves_logs(tmp_path):
    import run_workload_stage as stages
    from ovl_pipeline.canonical import file_hash
    t,remote,calls,job_file,_,worker_file,worker_root=staged(tmp_path)
    job=read_json(job_file);script=Path(job['argv'][1]);script.write_text('import time\nprint("retained terminal log",flush=True)\ntime.sleep(1)\n')
    job['required_files'][-1].update(bytes=script.stat().st_size,sha256=file_hash(script))
    job['export_roots']=[t.profile['remote_root']+'/record'];write_json(job_file,job);root=digest(job)
    marker=state(remote/'record/boundary-00000')
    with Journal(tmp_path/'j').lease() as j:
        h=Health(j,intent(),t.profile['pod_id'])
        hook=InitialRetention(t,h,job,root,selected(t),tmp_path/'initial',tmp_path/'health.json',tmp_path/'store')
        result=stages.run_stage(t,h,job_file,root,worker_file,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64,
                                sleep=lambda _:time.sleep(.05),initial_retention=hook)
        assert result['exit']['exit_code']==0 and h.jobs[root]['finished']
        assert (tmp_path/'initial/retained.json').exists()
        snapshot=read_json(tmp_path/'stage/export-001/export.json')
        assert set(snapshot['reused_paths'])=={'boundary-00000/checkpoint.json','boundary-00000/state.json','boundary-00000/state.safetensors'}
        assert 'retained terminal log' in (Path(result['exports'][0]['directory'])/'stdout.log').read_text()
        before=len(calls)
        assert stages.run_stage(t,h,job_file,root,worker_file,worker_root,tmp_path/'stage',tmp_path/'health.json',tmp_path/'stop.json','d'*64,
                                initial_retention=hook)==result
        assert len(calls)==before
