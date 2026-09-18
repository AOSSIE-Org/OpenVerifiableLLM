import pytest
from ovl_pipeline import runtime_activity as m
from ovl_pipeline.canonical import EvidenceError,read_json


def test_completed_update_observation_is_throttled_and_does_not_change_rng(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(m,'_last',None);monkeypatch.setattr(m,'_sequence',0);monkeypatch.setattr(m,'_process',None)
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    state=torch.get_rng_state().clone();c={'global_step':1,'cursor':32};m.update(c,clock=lambda:100)
    a=read_json(tmp_path/'activity.json');assert a['sequence']==1 and a['control']==c
    m.update({'global_step':2},clock=lambda:101);assert read_json(tmp_path/'activity.json')==a
    m.update({'global_step':3},clock=lambda:130);b=read_json(tmp_path/'activity.json')
    assert b['sequence']==2 and b['control']['global_step']==3 and b['process_instance']==a['process_instance']
    assert torch.equal(state,torch.get_rng_state()) and c=={'global_step':1,'cursor':32}


def test_telemetry_cannot_write_arbitrary_filename_or_follow_symlink(tmp_path,monkeypatch):
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'seed.key'))
    with pytest.raises(EvidenceError):m.update({})
    (tmp_path/'target').write_text('preserve');(tmp_path/'activity.json').symlink_to(tmp_path/'target')
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    with pytest.raises(EvidenceError):m.update({})
    assert (tmp_path/'target').read_text()=='preserve'


def test_telemetry_rejects_symlink_parent_and_propagates_failed_durable_write(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'_last',None)
    actual=tmp_path/'actual';actual.mkdir();alias=tmp_path/'alias';alias.symlink_to(actual,target_is_directory=True)
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(alias/'activity.json'))
    with pytest.raises(EvidenceError):m.update({})
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(actual/'activity.json'))
    def fail(*a):raise OSError('injected full disk')
    monkeypatch.setattr(m,'write_json',fail)
    with pytest.raises(OSError,match='full disk'):m.update({})
    assert not (actual/'activity.json').exists()
