"""Deterministic deadline checks; simulated durations give no throughput credit."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publish_progress_boundary as m
from ovl_pipeline.canonical import EvidenceError


def clock(monkeypatch):
    now=[1000.,100.]
    monkeypatch.setattr(m.time,'time',lambda:now[0])
    monkeypatch.setattr(m.time,'monotonic',lambda:now[1])
    return now


def test_slow_hook_allowance_uses_remaining_original_window(monkeypatch):
    now=clock(monkeypatch);calls=[]
    def execute(args,**kw):
        calls.append(kw['timeout']);now[0]+=300;now[1]+=300;return 'checked'
    run=m.deadline_command(1900,execute=execute)
    assert run(['git','commit'],timeout=600)=='checked'
    assert run(['git','push'],timeout=600)=='checked'
    assert calls==[600,600]
    # The same window is consumed, not renewed by either completed operation.
    assert m.deadline_command(1900,execute=lambda *a,**kw:kw['timeout'])(['git','status'])==120
    with pytest.raises(EvidenceError,match='exceeded'):
        run(['git','push'],timeout=600)
    assert calls[-1]==300
    with pytest.raises(EvidenceError,match='reached'):run(['git','status'])
    assert len(calls)==3


@pytest.mark.parametrize('wall_shift',[10000,-10000,0])
def test_wall_jump_or_elapsed_monotonic_never_renews_window(monkeypatch,wall_shift):
    now=clock(monkeypatch);calls=[]
    run=m.deadline_command(1100,execute=lambda *a,**kw:calls.append(kw))
    now[0]+=wall_shift;now[1]+=100
    with pytest.raises(EvidenceError,match='reached'):run(['git','commit'],timeout=600)
    assert calls==[]


def test_backward_clock_does_not_increase_command_budget(monkeypatch):
    now=clock(monkeypatch);run=m.deadline_command(1900,execute=lambda *a,**kw:kw['timeout'])
    now[0]-=1000;now[1]+=450
    assert run(['git','commit'],timeout=600)==450
    assert run(['git','status'])==120


def test_hook_failure_is_not_retried_or_converted_to_success(monkeypatch):
    clock(monkeypatch);calls=[]
    def denied(*a,**kw):calls.append(kw);raise EvidenceError('synthetic hook refusal')
    run=m.deadline_command(1900,execute=denied)
    with pytest.raises(EvidenceError,match='hook refusal'):run(['git','commit'],timeout=600)
    assert len(calls)==1
