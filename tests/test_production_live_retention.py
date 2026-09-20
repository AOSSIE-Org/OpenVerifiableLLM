"""Actual safe CPU states and local SSH substitute; no production execution."""
from pathlib import Path
import copy
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import production_live_retention as m
from test_pipeline import prepared
from test_production_health import bound
from test_production_chain import chain,resign
from test_workload_stage import intent
from production_health import ProductionHealth
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal


def test_actual_primary_snapshot_adopts_bytes_and_cannot_complete_job(prepared,tmp_path):
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path)
    # Select the existing complete signed prefix from the actual fixture path,
    # observed with the same bounded transport used by a live caller.
    import json
    chain_value=json.loads(record.read_live('chain.json',16*1024**2,int(time.time())+30))
    selection=m.record_selection(r,digest(r),chain_value);deadline=int(time.time())+60
    with Journal(tmp_path/'live-health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        out=tmp_path/'live-snapshot';args=(record,selection,digest(selection),root,h,tmp_path/'health.json',tmp_path/'live-objects',out,deadline,1024**2)
        receipt=m.retain(*args);assert receipt['result']=='PASS' and receipt['workload_complete'] is False
        assert not h.complete and not h.jobs[root]['finished']
        count=len(j.events);assert m.retain(*args)==receipt and len(j.events)==count
        # A primary snapshot cannot substitute for complete terminal retention.
        with pytest.raises((EvidenceError,FileNotFoundError)):
            h.job_exit(root,{'schema':'ovl.workload-job-exit.v1','job_sha256':root,'state':'EXITED','exit_code':0})
        path=Path(read_json(out/'snapshot-000/export.json')['files_directory'])/'state.safetensors'
        path.chmod(0o600);path.write_bytes(b'altered retained state')
        with pytest.raises(EvidenceError):m.retain(*args)


def test_interrupted_copy_preserved_with_one_retry_and_original_deadline(prepared,tmp_path,monkeypatch):
    control,transport,job,root,worker,r,bindings=bound(prepared,tmp_path)
    selection=m.record_selection(r,digest(r),read_json(tmp_path/'record/remote/chain.json'))
    deadline=int(time.time())+60;out=tmp_path/'retry-snapshot';original=m.export;attempts=[]
    def interrupted(*a,**kw):
        attempts.append(Path(a[3]));Path(a[3]).mkdir();(Path(a[3])/'partial').write_bytes(b'preserve these bytes')
        raise EvidenceError('explicit transfer interruption')
    with Journal(tmp_path/'retry-health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        args=[transport,selection,digest(selection),root,h,tmp_path/'health.json',tmp_path/'objects',out,deadline,1024**2]
        before=h.exported;monkeypatch.setattr(m,'export',interrupted)
        with pytest.raises(EvidenceError,match='interruption'):m.retain(*args)
        assert h.exported==before and not(out/'retention.json').exists()
        monkeypatch.setattr(m,'export',original);result=m.retain(*args)
        assert (out/'snapshot-000/partial').read_bytes()==b'preserve these bytes'
        assert 'snapshot-001' in result['receipt_path'] and len(attempts)==1
        args[-2]=deadline+1
        with pytest.raises(EvidenceError):m.retain(*args)


def replay_fixture():
    r,old,envelopes,key=chain();compatible={'explicit':'synthetic environment'}
    r['runtime']['compatible_environment_sha256']=digest(compatible);root=digest(r)
    for e in envelopes:e['body']['registration']=root
    envelopes[0]['body']['previous']=root;resign(envelopes,key)
    session={'schema':'ovl.production-replay-session.v1','registration_sha256':root,'chain_sha256':digest(envelopes),
        'code_root':r['code_root'],'process_observation':{'pid':42,'boot_id':'synthetic','start_ticks':'1'},
        'environment':{'compatible':compatible},'scope':'fresh-start-continuous-all-updates-numerical-replay',
        'prover_state_restored':False,'resume_supported':False}
    progress={'session_sha256':digest(session),'complete':False,'comparisons':[
        {'index':i,'boundary_sha256':digest(e),'state_root':e['body']['checkpoint']['state_root'],
         'control':e['body']['control'],'verifier_checkpoint':e['body']['checkpoint'],'result':'PASS'} for i,e in enumerate(envelopes[:2])]}
    return r,root,envelopes,session,progress


def test_replay_selection_is_bound_to_complete_registered_record_and_prefix():
    r,root,envelopes,session,progress=replay_fixture()
    selected=m.replay_selection(r,root,envelopes,session,progress)
    assert selected['path']=='verifier-boundary-00001' and selected['kind']=='replay-primary'
    assert selected['checkpoint']==envelopes[1]['body']['checkpoint']


@pytest.mark.parametrize('damage',['foreign-root','broken-record','session-parent','code','environment','restored-prover',
    'resume-claim','wrong-session','false-completion','missing-prefix','swapped-state','false-pass','boolean-index'])
def test_replay_metadata_mutations_fail_before_any_transfer(damage):
    r,root,envelopes,session,progress=replay_fixture()
    if damage=='foreign-root':root='f'*64
    elif damage=='broken-record':envelopes.pop(1)
    elif damage=='session-parent':session['chain_sha256']='e'*64
    elif damage=='code':session['code_root']='a'*64
    elif damage=='environment':session['environment']['compatible']['explicit']='other'
    elif damage=='restored-prover':session['prover_state_restored']=True
    elif damage=='resume-claim':session['resume_supported']=True
    elif damage=='wrong-session':progress['session_sha256']='e'*64
    elif damage=='false-completion':progress['complete']=True
    elif damage=='missing-prefix':progress['comparisons'].pop(0)
    elif damage=='swapped-state':progress['comparisons'][1]['verifier_checkpoint']={'schema':'ovl.checkpoint.v1',
        'state_root':'0'*64,'files':copy.deepcopy(progress['comparisons'][1]['verifier_checkpoint']['files'])}
    elif damage=='false-pass':progress['comparisons'][1]['result']='NOT_RUN'
    else:progress['comparisons'][0]['index']=False
    with pytest.raises(EvidenceError):m.replay_selection(r,root,envelopes,session,progress)


@pytest.mark.parametrize('mode',['record','replay'])
def test_actual_recovery_retains_complete_state_without_completion_credit(prepared,tmp_path,mode):
    from ovl_pipeline import training
    from ovl_pipeline.data import batches
    from ovl_pipeline.state import save_state
    control,transport,job,job_root,worker,r,bindings=bound(prepared,tmp_path)
    remote=tmp_path/'record/remote';chain_value=read_json(remote/'chain.json')
    chain_value={**chain_value,'complete':False,'boundaries':chain_value['boundaries'][:1]}
    directory,_=prepared;model,opt,c=training.initialize(r['recipe'])
    batch=next(batches(directory/'wikipedia',r['recipe']['context'],r['recipe']['batch_size']))
    c=training.update(model,opt,batch,c,r['coverage']['wikipedia']['targets'])
    cp=save_state(remote/'recovery-000000001',model,opt,c)
    recoveries={'registration_sha256':digest(r),'checkpoints':[{'path':'recovery-000000001','control':c,
        'checkpoint':cp,'last_primary_boundary_sha256':digest(chain_value['boundaries'][0]['body'])}]}
    selection=m.record_recovery_selection(r,digest(r),chain_value,recoveries)
    if mode=='replay':
        # Retention transport check only: a CPU-generated state stands in for a
        # verifier-owned recovery. Separate selector tests check replay ancestry.
        import shutil
        shutil.copytree(remote/selection['path'],remote/'verifier-recovery-000000001')
        selection={**selection,'kind':'replay-recovery','path':'verifier-recovery-000000001',
                   'control':{k:c[k] for k in ('phase','phase_step','global_step')}}
        value=read_json(job);value['kind']='full-replay';write_json(job,value);previous=job_root;job_root=digest(value)
        bindings={job_root:bindings[previous]}
    with Journal(tmp_path/'recovery-health').lease() as j:
        h=ProductionHealth(j,intent(),control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':job_root,'pod_id':h.pod,
                     'kind':'production-record' if mode=='record' else 'full-replay'})
        result=m.retain(transport,selection,digest(selection),job_root,h,tmp_path/'health.json',tmp_path/'objects',
                        tmp_path/'retained-recovery',int(time.time())+60,1024**2)
        assert result['result']=='PASS' and result['public_anchor']=='NOT_RUN' and not h.complete


def test_replay_recovery_selects_registered_step_and_marker_without_replay_credit():
    r,root,envelopes,session,progress=replay_fixture()
    cp=copy.deepcopy(envelopes[1]['body']['checkpoint']);cp['state_root']='a'*64
    recoveries={'session_sha256':digest(session),'recoveries':[{'global_step':40,'state_root':cp['state_root']}]}
    selection=m.replay_recovery_selection(r,root,envelopes,session,progress,recoveries,cp)
    assert selection['kind']=='replay-recovery' and selection['path']=='verifier-recovery-000000040'
    assert selection['control']=={'phase':'wikipedia','phase_step':40,'global_step':40}
    for changes in [{'global_step':30},{'global_step':41},{'global_step':1000},{'state_root':'b'*64}]:
        damaged=copy.deepcopy(recoveries);damaged['recoveries'][0].update(changes)
        with pytest.raises(EvidenceError):m.replay_recovery_selection(r,root,envelopes,session,progress,damaged,cp)
    damaged={**recoveries,'session_sha256':'f'*64}
    with pytest.raises(EvidenceError):m.replay_recovery_selection(r,root,envelopes,session,progress,damaged,cp)


@pytest.mark.parametrize('damage',['parent','duplicate','unregistered-step','primary-step','path','phase','cursor','root'])
def test_record_recovery_metadata_mutations(damage):
    r,root,envelopes,key=chain();primary={'schema':'ovl.production-chain.v1','complete':False,'boundaries':envelopes[:1]}
    c={**envelopes[0]['body']['control'],'global_step':10,'phase_step':10,'cursor':480}
    cp=copy.deepcopy(envelopes[1]['body']['checkpoint'])
    item={'path':'recovery-000000010','control':c,'checkpoint':cp,'last_primary_boundary_sha256':digest(envelopes[0]['body'])}
    recoveries={'registration_sha256':root,'checkpoints':[item]}
    assert m.record_recovery_selection(r,root,primary,recoveries)['kind']=='record-recovery'
    if damage=='parent':item['last_primary_boundary_sha256']='f'*64
    elif damage=='duplicate':recoveries['checkpoints'].append(copy.deepcopy(item))
    elif damage=='unregistered-step':c.update(global_step=11,phase_step=11)
    elif damage=='primary-step':c.update(global_step=30,phase_step=30)
    elif damage=='path':item['path']='../recovery-000000010'
    elif damage=='phase':c['phase']='conversation'
    elif damage=='cursor':c['cursor']=0
    else:recoveries['registration_sha256']='e'*64
    with pytest.raises(EvidenceError):m.record_recovery_selection(r,root,primary,recoveries)
