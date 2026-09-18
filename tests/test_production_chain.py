"""Real local Ed25519 signatures over synthetic state declarations, no training."""
import copy
import pytest
from nacl.signing import SigningKey
from test_production_contract import registration
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
