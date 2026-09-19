"""Compare full independent cursor census with actual batch target IDs, no training."""
import pytest
from test_pipeline import prepared
from ovl_pipeline.canonical import EvidenceError,digest
from ovl_pipeline.fixture import recipe
from ovl_pipeline.data import batches
from ovl_pipeline.production_cursors import boundary_cursors

@pytest.mark.parametrize('phase',['wikipedia','conversation'])
@pytest.mark.parametrize('context,batch',[(1,1),(4,3),(16,3),(64,7)])
def test_full_cursor_map_matches_actual_batches_and_tails(prepared,phase,context,batch):
    root,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size']);r.update(context=context,batch_size=batch);r['model']['max_seq_len']=context
    count=0;cursor=0;expected=[{'step':0,'windows':0,'targets':0}];windows=0
    for b in batches(root/phase,context,batch):
        ids=b['target_ids'][b['mask']].tolist()
        assert ids==list(range(cursor,cursor+len(ids)))
        count+=1;cursor+=len(ids);windows+=b['inputs'].shape[0]
        expected.append({'step':count,'windows':windows,'targets':cursor})
    got=boundary_cursors(root/phase,r,digest(manifest['streams'][phase]),list(range(count+1)))
    assert got['boundaries']==expected and got['targets']==cursor and got['training_coverage']=='NOT_RUN'
    selected=sorted({0,count//2,count})
    sparse=boundary_cursors(root/phase,r,digest(manifest['streams'][phase]),selected)
    assert sparse['boundaries']==[expected[n] for n in selected]
    assert sparse['documents']==got['documents'] and sparse['targets']==got['targets']

@pytest.mark.parametrize('change',['wrong-root','missing-tail','extra-step','duplicate','out-of-order','empty','boolean'])
def test_cursor_census_refuses_untrusted_or_incomplete_schedule(prepared,change):
    root,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size']);stream=digest(manifest['streams']['wikipedia'])
    count=sum(1 for _ in batches(root/'wikipedia',r['context'],r['batch_size']))
    steps=list(range(count+1))
    if change=='wrong-root':stream='0'*64
    elif change=='missing-tail':steps.pop()
    elif change=='extra-step':steps.append(count+1)
    elif change=='duplicate':steps.append(count)
    elif change=='out-of-order':steps.reverse()
    elif change=='empty':steps=[]
    else:steps[0]=False
    with pytest.raises(EvidenceError):boundary_cursors(root/'wikipedia',r,stream,steps)
