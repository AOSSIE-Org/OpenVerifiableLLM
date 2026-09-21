"""Actual safe CPU files and local transport; no live CUDA or production admission."""
from pathlib import Path
import copy
import time
import pytest
import production_checkpoint_poll as m
import production_live_retention as live
from production_health import ProductionHealth
from test_production_health import bound
from test_pipeline import prepared
from test_workload_stage import intent
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal


def policy():return {'schema':'ovl.production-checkpoint-copy-policy.v1','export_seconds':60,'maximum_checkpoint_bytes':1024**2}


def test_complete_primary_poll_recovery_keeps_original_copy_clock_and_export_identity(prepared,tmp_path):
    control,t,job,root,worker,r,bindings=bound(prepared,tmp_path);w=intent()
    def hook(h):return m.CheckpointRetention(r,root,h,tmp_path/'health.json',t,tmp_path/'objects',tmp_path/'poll',policy())
    with Journal(tmp_path/'health').lease() as journal:
        h=ProductionHealth(journal,w,control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        a=hook(h).poll();saved=read_json(Path(a['directory'])/'copy-intent.json');count=len(journal.events)
        b=hook(h).poll();assert a==b and len(journal.events)==count
        assert not h.complete and not h.jobs[root]['finished']
    with Journal(tmp_path/'health').lease() as journal:
        h=ProductionHealth(journal,w,control.profile['pod_id'],r,bindings);assert hook(h).poll()==a
        assert read_json(Path(a['directory'])/'copy-intent.json')==saved
        receipt=read_json(Path(a['retention']['receipt_path']));p=Path(receipt['files_directory'])/'state.safetensors'
        p.chmod(0o600);p.write_bytes(b'changed')
        with pytest.raises(EvidenceError):hook(h).poll()


@pytest.mark.parametrize('damage',['parent','signature','missing-state','empty-recoveries','foreign-recoveries','renew-clock','regression'])
def test_invalid_observation_or_recovery_never_renews_export(prepared,tmp_path,damage):
    control,t,job,root,worker,r,bindings=bound(prepared,tmp_path);remote=tmp_path/'record/remote'
    with Journal(tmp_path/'health').lease() as journal:
        h=ProductionHealth(journal,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        hook=m.CheckpointRetention(r,root,h,tmp_path/'health.json',t,tmp_path/'objects',tmp_path/'poll',policy())
        if damage=='regression':
            hook.poll();value=read_json(remote/'chain.json');value['boundaries']=value['boundaries'][:1]
            value['complete']=False;write_json(remote/'chain.json',value)
        elif damage=='renew-clock':
            value=hook.poll();p=Path(value['directory'])/'copy-intent.json';saved=read_json(p);saved['deadline_epoch']+=1;write_json(p,saved)
        elif damage in ('parent','signature'):
            chain=read_json(remote/'chain.json');chain['boundaries'][-1]['body']['registration']='e'*64 if damage=='parent' else chain['boundaries'][-1]['body']['registration']
            if damage=='signature':chain['boundaries'][-1]['signature']='f'*128
            write_json(remote/'chain.json',chain)
        elif damage=='missing-state':
            chain=read_json(remote/'chain.json');(remote/chain['boundaries'][-1]['body']['checkpoint_path']/'state.safetensors').unlink()
        elif damage=='empty-recoveries':write_json(remote/'recoveries.json',{'registration_sha256':digest(r),'checkpoints':[]})
        else:write_json(remote/'recoveries.json',{'registration_sha256':'f'*64,'checkpoints':[{}]})
        before=len([e for e in journal.events if e['body'].get('advances_export')])
        with pytest.raises((EvidenceError,FileNotFoundError)):hook.poll()
        assert len([e for e in journal.events if e['body'].get('advances_export')])==before


def test_observed_copy_failure_has_one_preserved_retry_and_never_new_deadline(prepared,tmp_path,monkeypatch):
    from pod_transfer import TransientTransportError
    control,t,job,root,worker,r,bindings=bound(prepared,tmp_path)
    with Journal(tmp_path/'health').lease() as journal:
        h=ProductionHealth(journal,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        hook=m.CheckpointRetention(r,root,h,tmp_path/'health.json',t,tmp_path/'objects',tmp_path/'poll',policy())
        original=t.get;partials=[]
        def fail(name,destination,*args,**kw):
            partial=destination.with_name(destination.name+'.partial');partial.write_bytes(b'');partials.append(partial)
            error=TransientTransportError('explicit copy interruption');error.transfer_counts={'bytes_sent':0,'bytes_received':0};raise error
        monkeypatch.setattr(t,'get',fail)
        with pytest.raises(EvidenceError):hook.poll()
        selected=next((tmp_path/'poll/checkpoints').iterdir());old=read_json(selected/'copy-intent.json')
        monkeypatch.setattr(t,'get',original);a=hook.poll()
        assert read_json(selected/'copy-intent.json')==old
        assert len(partials)==1 and partials[0].read_bytes()==b''
        assert 'snapshot-001' in a['retention']['receipt_path']


def test_actual_fresh_cpu_replay_outputs_are_retained_under_selected_record(prepared,cpu_runtime,tmp_path,monkeypatch):
    import shutil
    from test_production_replay import setup
    control,t,job,old,worker,_,bindings=bound(prepared,tmp_path)
    numerical=tmp_path/'fresh-replay';numerical.mkdir()
    r,envelopes,_,prover,run=setup(prepared,numerical,monkeypatch)
    # Seed from the actual record, never from verifier output. Production
    # completely retains this tree before starting the separate replay process.
    from pod_versioned_export import export
    shutil.copytree(prover,tmp_path/'record/remote/complete-record')
    export(t,'complete-record',tmp_path/'objects',tmp_path/'record-cache',int(time.time())+60)
    output=numerical/'verifier';report=run(output);assert report['result']=='PASS'
    shutil.copytree(output,tmp_path/'record/remote',dirs_exist_ok=True)
    def no_payload_transfer(*args,**kwargs):raise AssertionError('fresh replay must reuse retained record bytes')
    t.get=no_payload_transfer
    value=read_json(job);value['kind']='full-replay';write_json(job,value);root=digest(value);bindings={root:bindings[old]}
    with Journal(tmp_path/'health').lease() as journal:
        h=ProductionHealth(journal,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'full-replay'})
        hook=m.CheckpointRetention(r,root,h,tmp_path/'health.json',t,tmp_path/'objects',tmp_path/'poll',policy(),envelopes=envelopes)
        result=hook.poll();assert result['selection']['kind']=='replay-primary'
        assert result['selection']['control']==envelopes[-1]['body']['control']
        assert result['retention']['numerical_replay']=='NOT_RUN' and not h.complete
        progress=read_json(tmp_path/'record/remote/progress.json');progress['session_sha256']='e'*64
        write_json(tmp_path/'record/remote/progress.json',progress)
        with pytest.raises(EvidenceError):hook.poll()


from test_gpu_pilot import cpu_runtime


def test_all_actual_primary_steps_include_equal_step_base_to_chat_transition(prepared,tmp_path):
    control,t,job,root,worker,r,bindings=bound(prepared,tmp_path);remote=tmp_path/'record/remote'
    full=read_json(remote/'chain.json')['boundaries'];steps=[]
    with Journal(tmp_path/'health').lease() as journal:
        h=ProductionHealth(journal,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        for i,env in enumerate(full):
            write_json(remote/'chain.json',{'schema':'ovl.production-chain.v1','complete':False,'boundaries':full[:i+1]})
            hook=m.CheckpointRetention(r,root,h,tmp_path/'health.json',t,tmp_path/'objects',tmp_path/'poll',policy())
            result=hook.poll();assert result['selection']['control']==env['body']['control']
            steps.append((result['selection']['control']['global_step'],result['selection']['control']['phase']))
        assert any(a[0]==b[0] and a[1]=='wikipedia' and b[1]=='conversation' for a,b in zip(steps,steps[1:]))
        assert len(h.exports)==len(full) and not h.complete
