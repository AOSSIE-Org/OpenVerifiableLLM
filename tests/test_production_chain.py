"""Real local Ed25519 signatures over synthetic state declarations, no training."""
import copy
import pytest
from nacl.signing import SigningKey
from test_production_contract import registration
from test_pipeline import prepared
from ovl_pipeline.canonical import EvidenceError,digest,sha256
from ovl_pipeline.production_chain import schedule,verify_chain
from ovl_pipeline.training import signed


def chain():
    r=registration();key=SigningKey.generate();r['run_public_key']=bytes(key.verify_key).hex();root=digest(r)
    result=[];previous=root;transcript=sha256(b'ovl.batch-transcript.v1')
    for item in schedule(r):
        if item['kind'] not in ('initial','transition'):transcript=digest({'synthetic-step':item['global_step']})
        c={k:item[k] for k in ('phase','phase_step','global_step')}
        c.update(cursor=item['phase_step']*48,transcript=transcript,schedule='constant-lr-v1',accumulation='none',scaler='none')
        checkpoint={'schema':'ovl.checkpoint.v1','state_root':r['initialization']['state_sha256'] if item['kind']=='initial' else digest(item),
                    'files':[{'path':n,'bytes':1,'sha256':'0'*64} for n in ('state.json','state.safetensors')]}
        b={'schema':'ovl.production-boundary.v1','index':item['index'],'registration':root,'previous':previous,
           'kind':item['kind'],'control':c,'checkpoint_path':f"boundary-{item['index']:05d}",'checkpoint':checkpoint}
        result.append(signed(b,key));previous=digest(b)
    return r,root,result,key


def resign(values,key):
    for index,env in enumerate(values):
        if index:env['body']['previous']=digest(values[index-1]['body'])
        values[index]=signed(env['body'],key)


def test_exact_two_phase_schedule_and_scoped_signatures():
    r,root,values,key=chain();result=verify_chain(r,root,values,complete=True)
    assert len(values)==70 and result['complete_schedule_checked'] is True
    assert [b['body']['kind'] for b in values].count('transition')==1
    assert all(result[k]=='NOT_RUN' for k in ('checkpoint_bytes','public_boundary_anchors','training_replay','production_admission'))
    assert verify_chain(r,root,values[:2],complete=False)['complete_schedule_checked'] is False
    with pytest.raises(EvidenceError):verify_chain(r,root,values[:2],complete=True)

@pytest.mark.parametrize('mutation',['wrong-key','foreign-root','missing','duplicate','reorder','wrong-parent','wrong-step','phase-regression',
                                    'missing-tail','early-tail','transition-transcript','warm-init','unsafe-path','pickle','duplicate-file','oversized','empty'])
def test_forged_signed_and_relinked_declarations_rejected(mutation):
    r,root,v,key=chain();transition=next(i for i,e in enumerate(v) if e['body']['kind']=='transition')
    if mutation=='wrong-key':r['run_public_key']=bytes(SigningKey.generate().verify_key).hex();root=digest(r)
    elif mutation=='foreign-root':root='0'*64
    elif mutation=='missing':v.pop(2)
    elif mutation=='duplicate':v.insert(2,copy.deepcopy(v[1]))
    elif mutation=='reorder':v[1],v[2]=v[2],v[1]
    elif mutation=='wrong-parent':v[0]['body']['previous']='0'*64
    elif mutation=='wrong-step':v[2]['body']['control']['phase_step']+=1
    elif mutation=='phase-regression':v[transition+1]['body']['control']['phase']='wikipedia'
    elif mutation=='missing-tail':v[-1]['body']['control']['cursor']-=1
    elif mutation=='early-tail':v[1]['body']['control']['cursor']=r['coverage']['wikipedia']['targets']
    elif mutation=='transition-transcript':v[transition]['body']['control']['transcript']='0'*64
    elif mutation=='warm-init':v[0]['body']['checkpoint']['state_root']='0'*64
    elif mutation=='unsafe-path':v[1]['body']['checkpoint_path']='../state'
    elif mutation=='pickle':v[1]['body']['checkpoint']['files'][1]['path']='state.pkl'
    elif mutation=='duplicate-file':v[1]['body']['checkpoint']['files'][1]=copy.deepcopy(v[1]['body']['checkpoint']['files'][0])
    elif mutation=='oversized':v[1]['body']['checkpoint']['files'][1]['bytes']=2*1024**3
    else:v=[]
    resign(v,key)
    with pytest.raises(EvidenceError):verify_chain(r,root,v,complete=True)


def actual_artifacts(prepared,tmp_path):
    """Actual tiny CPU updates and safe checkpoints; invented production parents."""
    from ovl_pipeline import training
    from ovl_pipeline.coverage import schedule_counts
    from ovl_pipeline.data import batches
    from ovl_pipeline.fixture import recipe
    from ovl_pipeline.state import capture,save_state,state_root
    from ovl_pipeline.production_contract import checkpoint_count
    directory,manifest=prepared;r=registration();r['recipe']=recipe(manifest['tokenizer']['vocab_size'])
    r['recipe']['boundary_every']=3;r['recovery_every']=1
    key=SigningKey.generate();r['run_public_key']=bytes(key.verify_key).hex()
    for phase in ('wikipedia','conversation'):
        c=schedule_counts(directory/phase,r['recipe']);r['coverage'][phase]=c
        f=r['forecast_input']['phases'][phase]
        f.update(updates=c['updates'],recipe_sha256=digest(r['recipe']),stream_sha256=c['stream_sha256'],
                 schedule_sha256=digest(c),production_checkpoint_every=1,production_checkpoints=c['updates'],
                 measured_checkpoints=100,measured_checkpoint_every=1)
    model,opt,control=training.initialize(r['recipe']);r['initialization']['state_sha256']=state_root(*capture(model,opt,control))
    root=digest(r);values=[];previous=root;out=tmp_path/'checkpoints';out.mkdir()
    def boundary(kind):
        nonlocal previous
        path=f'boundary-{len(values):05d}';m=save_state(out/path,model,opt,control)
        b={'schema':'ovl.production-boundary.v1','index':len(values),'registration':root,'previous':previous,
           'kind':kind,'control':control.copy(),'checkpoint_path':path,'checkpoint':m}
        values.append(signed(b,key));previous=digest(b)
    boundary('initial')
    for phase in ('wikipedia','conversation'):
        if phase=='conversation':opt,control=training._transition(model,r['recipe'],control);boundary('transition')
        for batch in batches(directory/phase,r['recipe']['context'],r['recipe']['batch_size']):
            control=training.update(model,opt,batch,control,r['coverage'][phase]['targets'])
            if control['phase_step']==r['coverage'][phase]['updates']:boundary('base' if phase=='wikipedia' else 'final')
            elif control['phase_step']%r['recipe']['boundary_every']==0:boundary('progress')
    return r,root,values,key,out,{'wikipedia':directory/'wikipedia','conversation':directory/'conversation'}


def test_actual_full_input_cursors_and_safe_checkpoint_bytes(prepared,tmp_path):
    from ovl_pipeline.production_chain import verify_artifacts
    r,root,v,key,out,streams=actual_artifacts(prepared,tmp_path)
    result=verify_artifacts(r,root,v,out,streams,complete=True)
    assert result['checkpoint_bytes']=='PASS' and result['full_stream_census']=='PASS'
    assert result['training_replay']=='NOT_RUN' and result['public_boundary_anchors']=='NOT_RUN'


@pytest.mark.parametrize('change',['intermediate-cursor','checkpoint-bytes','checkpoint-control','stream','extra-file'])
def test_artifact_checker_rejects_relinked_false_cursors_and_altered_state(prepared,tmp_path,change):
    from ovl_pipeline.production_chain import verify_artifacts
    from ovl_pipeline.canonical import write_json
    from ovl_pipeline.state import read_state,unpack,pack,state_root
    from ovl_pipeline.canonical import inventory
    from safetensors.torch import save_file
    r,root,v,key,out,streams=actual_artifacts(prepared,tmp_path)
    b=v[1]['body'];path=out/b['checkpoint_path']
    if change=='intermediate-cursor':b['control']['cursor']-=1;resign(v,key)
    elif change=='checkpoint-bytes':(path/'state.safetensors').write_bytes(b'altered')
    elif change=='extra-file':(path/'unsafe.pkl').write_bytes(b'extra')
    elif change=='stream':streams['wikipedia']=streams['conversation']
    else:
        md,ts=read_state(path,b['checkpoint']);obj=unpack(md['tree'],ts);obj['control']['cursor']-=1
        tensors={};tree=pack(obj,tensors)
        from ovl_pipeline.state import tensor_digest
        md={'schema':'ovl.state.v1','tree':tree,'tensor_root':tensor_digest(tensors)}
        save_file(tensors,str(path/'state.safetensors'));write_json(path/'state.json',md)
        b['checkpoint']={'schema':'ovl.checkpoint.v1','state_root':state_root(md,tensors),'files':inventory(path,['state.json','state.safetensors'])}
        write_json(path/'checkpoint.json',b['checkpoint']);resign(v,key)
    with pytest.raises(EvidenceError):verify_artifacts(r,root,v,out,streams,complete=True)
