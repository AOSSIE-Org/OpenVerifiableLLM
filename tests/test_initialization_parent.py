"""Real tiny CPU states; explicit CPU/process substitutes, no CUDA acceptance."""
import pytest

from initialization_parent import check,check_verification
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from test_initialization import distinct_process_observations
from ovl_pipeline import initialization
from ovl_pipeline.fixture import recipe
from ovl_pipeline.canonical import EvidenceError,digest,inventory,write_json


def files(path):return inventory(path,[p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()])


@pytest.fixture
def cycle(cpu_runtime,distinct_process_observations,prepared,tmp_path):
    data,manifest=prepared;record_dir=tmp_path/'record';verification_dir=tmp_path/'verify'
    r=initialization.record(data/'wikipedia',recipe(manifest['tokenizer']['vocab_size']),
        {'schema':'ovl.gpu-kernel.v1','precision':'bf16'},record_dir,warmup_updates=2)
    v=initialization.verify(data/'wikipedia',record_dir,digest(r),verification_dir)
    binding={'schema':'ovl.initialization-record-parent-binding.v1',
        **{k+'_sha256':digest(r[k]) for k in ('recipe','kernel')},
        **{k:r[k] for k in ('stream_sha256','code_root','parameter_count','warmup_updates')}}
    return record_dir,verification_dir,r,v,binding


def test_complete_record_and_regeneration_retained_under_external_binding(cycle):
    rd,vd,r,v,b=cycle
    parent=check(rd,files(rd),b)
    assert parent['initial_state_sha256']==r['checkpoint']['state_root']
    checked=check_verification(vd,files(vd),rd,files(rd),b)
    assert checked['verification_sha256']==digest(v) and checked['production_admission']=='NOT_RUN'
    assert checked['independent_third_party'] is False


@pytest.mark.parametrize('damage',['bytes','omitted','marker','control','recipe','kernel','code_root',
    'parameter_count','warmup_updates','warmup_bool','warmup_retained','report_scope','symlink','extra_state'])
def test_altered_or_unselected_initializer_parent_refused(cycle,damage,tmp_path):
    rd,vd,r,v,b=cycle;selected=files(rd)
    if damage=='bytes':(rd/'initial-state/state.safetensors').write_bytes(b'altered')
    elif damage=='omitted':selected=[f for f in selected if f['path']!='initial-state/state.safetensors']
    elif damage=='symlink':
        target=tmp_path/'saved-state';p=rd/'initial-state/state.safetensors';p.rename(target);p.symlink_to(target)
    elif damage=='extra_state':(rd/'initial-state/extra').write_text('extra');selected=files(rd)
    elif damage in ('recipe','kernel','code_root'):b[damage+'_sha256' if damage!='code_root' else damage]='f'*64
    elif damage in ('parameter_count','warmup_updates'):b[damage]+=1
    else:
        if damage=='marker':write_json(rd/'initial-state/checkpoint.json',{'schema':'wrong'})
        elif damage=='control':r['control']['transcript']='f'*64
        elif damage=='warmup_bool':r['warmup_updates']=True
        elif damage=='warmup_retained':r['warmup_weights_discarded']=False
        else:r['scope']='production'
        write_json(rd/'record.json',r);selected=files(rd)
    with pytest.raises((EvidenceError,FileNotFoundError)):check(rd,selected,b)


@pytest.mark.parametrize('damage',['same_process','state','record','recipe','stream','code','warmup',
    'runtime','restored','third_party','scope','distinct_bool','omitted','parent_changed'])
def test_regeneration_report_cannot_substitute_missing_state_or_promote_scope(cycle,damage):
    rd,vd,r,v,b=cycle;record_files=files(rd)
    if damage=='same_process':v['process_observation']=r['process_observation']
    elif damage in ('state','record','recipe','stream','code'):
        key={'state':'initial_state_sha256','record':'record_sha256','recipe':'recipe_sha256','stream':'stream_sha256','code':'code_root'}[damage];v[key]='f'*64
    elif damage=='warmup':v['warmup_updates']+=1
    elif damage=='runtime':v['environment']['compatible']={'changed':True}
    elif damage=='restored':v['prover_tensors_loaded_as_state']=True
    elif damage=='third_party':v['independent_third_party']=True
    elif damage=='scope':v['scope']='all-training-replayed'
    elif damage=='distinct_bool':v['distinct_process_from_record']=1
    elif damage=='parent_changed':(rd/'initial-state/state.safetensors').write_bytes(b'changed after parent check')
    write_json(vd/'verification.json',v);selected=files(vd)
    if damage=='omitted':selected=[]
    with pytest.raises((EvidenceError,FileNotFoundError)):
        check_verification(vd,selected,rd,record_files,b)
