"""Real CPU fixture states; no CUDA execution or third-party verification claim."""
from pathlib import Path
import copy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import verify_pilot_cycle as m
from test_gpu_pilot import cpu_runtime,prepared
from ovl_pipeline import gpu_pilot,training
from ovl_pipeline.fixture import recipe
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json


@pytest.fixture
def cycle(cpu_runtime,prepared,tmp_path):
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size'])
    kernel={'schema':'ovl.gpu-kernel.v1','precision':'fp32'}
    record=gpu_pilot.record(directory/'wikipedia',r,kernel,tmp_path/'record',updates=6,warmup_updates=1,checkpoint_every=2)
    gpu_pilot.replay(directory/'wikipedia',tmp_path/'record',digest(record),tmp_path/'replay')
    gpu_pilot.replay(directory/'wikipedia',tmp_path/'record',digest(record),tmp_path/'resume',resume_from=2)
    selected={'binding':{'schema':'ovl.pilot-record-parent-binding.v1','recipe_sha256':digest(r),'kernel_sha256':digest(kernel),
                         'stream_sha256':digest(manifest['streams']['wikipedia']),'code_root':training.code_root()},
              'expected_record':digest(record),'resume_from':2}
    for name in ('record','replay','resume'):
        root=tmp_path/name;selected[name+'_directory']=root
        selected[name+'_files']=inventory(root,[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()])
    return selected


def test_complete_actual_states_and_scope(cycle):
    result=m.verify(**cycle)
    assert result['record_safe_states']==4
    assert result['full_replay']['checked_boundaries']==[0,1,2,3]
    assert result['resume_probe']['checked_boundaries']==[0,2,3]
    assert result['resume_probe']['updates_reported_recomputed']==2
    assert result['independent_third_party'] is False and result['production_acceptance']=='NOT_RUN'
    assert 'executes no numerical replay' in result['scope']


@pytest.mark.parametrize('damage',['record-root','binding','state-byte','missing-state','extra-file','symlink','missing-comparison',
    'report-parent','resume-as-full','wrong-resume','targets','environment','independent-claim','checkpoint-count','changed-settings',
    'boolean-index','boolean-count'])
def test_reject_broken_retained_cycles_even_with_rehashed_outer_inventory(cycle,damage):
    s=copy.deepcopy(cycle);root=Path(s['replay_directory'])
    if damage=='record-root':s['expected_record']='a'*64
    elif damage=='binding':s['binding']['recipe_sha256']='b'*64
    elif damage=='state-byte':
        p=root/'verifier-boundary-00001/state.safetensors';b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
    elif damage=='missing-state':(root/'verifier-boundary-00001/state.safetensors').unlink()
    elif damage=='extra-file':(root/'hidden').write_text('unselected')
    elif damage=='symlink':(root/'link').symlink_to(root/'verification.json')
    elif damage=='changed-settings':
        p=Path(s['record_directory'])/'settings.json';v=read_json(p);v['warmup_updates']+=1;write_json(p,v)
    else:
        p=root/'verification.json';v=read_json(p)
        if damage=='missing-comparison':v['compared'].pop(1)
        elif damage=='report-parent':v['record_sha256']='e'*64
        elif damage=='resume-as-full':v['scope']='training-resume-continuation-probe'
        elif damage=='wrong-resume':s['resume_from']=1
        elif damage=='targets':v['measured_targets']-=1
        elif damage=='environment':v['environment']['compatible']={'different':True}
        elif damage=='independent-claim':v['independent_third_party']=True
        elif damage=='checkpoint-count':v['verifier_checkpoints_saved']-=1
        elif damage=='boolean-index':v['compared'][0]['index']=False
        elif damage=='boolean-count':v['updates_recomputed']=True
        write_json(p,v)
    # An attacker supplying matching transport hashes still cannot change the
    # independently selected record ancestry or actual safe-state identities.
    if damage!='symlink':
        for name in ('record','replay','resume'):
            root=Path(s[name+'_directory'])
            s[name+'_files']=inventory(root,[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()])
    with pytest.raises((EvidenceError,FileNotFoundError)):m.verify(**s)
