"""Timing never supplies replay credit; CPU fixtures do not qualify CUDA."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

import pytest
from ovl_pipeline import gpu_pilot, phase_timing as t
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.fixture import recipe
from test_gpu_pilot import cpu_runtime
from test_pipeline import prepared
from verify_pilot_cycle import replay_states


def test_exclusive_categories_context_cleanup_and_parent_binding():
    ticks=iter([100,130,140,180]);p=t.Collector(clock=lambda:next(ticks))
    with p.activate():
        with t.observe('one'):
            with pytest.raises(EvidenceError,match='overlapping'):
                with t.observe('two'):pass
        t.phase('measured')
        with t.observe('one'):pass
    # Outside the explicit scope no clock or CUDA operation is performed.
    with t.observe('ignored',cuda=True):pass
    value=p.report({'parent':1},scope='fixture')
    assert [e['wall_ns'] for e in value['measurements']]==[40,30]
    assert t.validate(value,{'parent':1},scope='fixture')==value
    bad=copy.deepcopy(value);bad['parent_sha256']='0'*64
    with pytest.raises(EvidenceError,match='parent'):t.validate(bad,{'parent':1},scope='fixture')
    bad=copy.deepcopy(value);bad['measurements']*=2
    with pytest.raises(EvidenceError,match='duplicate'):t.validate(bad,{'parent':1},scope='fixture')
    bad=copy.deepcopy(value);bad['measurements'][0]['cuda_stream_span_ns']=1
    with pytest.raises(EvidenceError,match='without events'):t.validate(bad,{'parent':1},scope='fixture')


def test_cuda_event_span_is_separate_from_wall_and_not_a_busy_counter(monkeypatch):
    import torch
    calls=[]
    class Event:
        def __init__(self,*,enable_timing):assert enable_timing
        def record(self):calls.append('record')
        def synchronize(self):calls.append('synchronize')
        def elapsed_time(self,other):return 0.5
    monkeypatch.setattr(torch.cuda,'Event',Event)
    ticks=iter([0,900000]);p=t.Collector(clock=lambda:next(ticks))
    with p.measure('numerical',cuda=True):pass
    value=p.report({},scope='fixture');entry=value['measurements'][0]
    assert entry['wall_ns']==900000 and entry['cuda_stream_span_ns']==500000
    assert calls==['record','record','synchronize']
    assert 'not kernel-busy' in value['cuda_interpretation']


@pytest.mark.parametrize('phase',['wikipedia','conversation'])
def test_profiled_cli_preserves_every_state_and_complete_replay(cpu_runtime,prepared,tmp_path,monkeypatch,phase):
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size']);kernel={'schema':'ovl.gpu-kernel.v1','precision':'fp32'}
    reference=gpu_pilot.record(directory/phase,r,kernel,tmp_path/'reference',updates=5,warmup_updates=2,checkpoint_every=2)
    write_json(tmp_path/'recipe.json',r);write_json(tmp_path/'kernel.json',kernel)
    monkeypatch.setattr(sys,'argv',['gpu_pilot','record','--stream',str(directory/phase),'--recipe',str(tmp_path/'recipe.json'),
        '--kernel',str(tmp_path/'kernel.json'),'--output',str(tmp_path/'record'),'--updates','5','--warmup-updates','2',
        '--checkpoint-every','2','--profile-timing'])
    assert gpu_pilot.main()==0
    recorded=read_json(tmp_path/'record/record.json');profile=read_json(tmp_path/'record/timing.json')
    assert [b['checkpoint'] for b in recorded['boundaries']]==[b['checkpoint'] for b in reference['boundaries']]
    assert (tmp_path/'record/updates.jsonl').read_bytes()==(tmp_path/'reference/updates.jsonl').read_bytes()
    t.validate(profile,recorded,scope='operator-pilot-phase-timing')
    measured={e['operation']:e for e in profile['measurements'] if e['phase']=='measured'}
    assert measured['numerical_update']['calls']==measured['batch_preparation']['calls']==5
    assert measured['checkpoint_serialization']['calls']==3
    monkeypatch.setattr(sys,'argv',['gpu_pilot','replay','--stream',str(directory/phase),'--record-directory',str(tmp_path/'record'),
        '--expected-record-sha256',digest(recorded),'--output',str(tmp_path/'replay'),'--profile-timing'])
    assert gpu_pilot.main()==0
    root=tmp_path/'replay'
    def files():return inventory(root,[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()])
    assert replay_states(root,files(),recorded,digest(recorded),resume_from=None)['actual_safe_states']==4
    bad=read_json(root/'timing.json');bad['parent_sha256']='0'*64;write_json(root/'timing.json',bad)
    with pytest.raises(EvidenceError,match='timing parent'):replay_states(root,files(),recorded,digest(recorded),resume_from=None)
