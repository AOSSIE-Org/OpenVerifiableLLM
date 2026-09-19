"""Exact original scan bodies, actual tiny files, explicit deterministic clocks."""
import itertools
from types import SimpleNamespace
import threading

import pytest
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from ovl_pipeline import production_observation as m,data,coverage,production_cursors,runtime_activity
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.fixture import recipe


@pytest.fixture
def observations(tmp_path,monkeypatch):
    events=[];ticks=itertools.count(step=31)
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    monkeypatch.setattr(m,'time',SimpleNamespace(monotonic=lambda:next(ticks)))
    monkeypatch.setattr(m,'_pass_index',0)
    for key,value in (('_last',None),('_sequence',0),('_process',None)):monkeypatch.setattr(runtime_activity,key,value)
    original=runtime_activity._emit
    def emit(value,**kw):
        original(value,**kw);events.append(read_json(tmp_path/'activity.json'))
    monkeypatch.setattr(runtime_activity,'_emit',emit)
    return events


@pytest.mark.parametrize('phase',['wikipedia','conversation'])
def test_nested_original_scans_equal_unobserved_results_without_global_changes(prepared,observations,phase):
    directory,manifest=prepared;directory=directory/phase;stream=read_json(directory/'stream.json');r=recipe(manifest['tokenizer']['vocab_size'])
    before=(data.rows,data.validate_stream,coverage.rows,coverage.validate_stream,production_cursors.rows,production_cursors.validate_stream)
    assert m.validate_stream(directory,stream)==data.validate_stream(directory,stream)
    census=coverage.schedule_counts(directory,r);assert m.schedule_counts(directory,r)==census
    steps=list(range(census['updates']+1))
    assert m.boundary_cursors(directory,r,digest(stream),steps)==production_cursors.boundary_cursors(directory,r,digest(stream),steps)
    assert before==(data.rows,data.validate_stream,coverage.rows,coverage.validate_stream,production_cursors.rows,production_cursors.validate_stream)
    complete=[e for e in observations if e['complete']]
    assert [e['operation'] for e in complete]==['stream-validation','stream-validation','coverage-census','stream-validation','boundary-cursor-census']
    assert [e['pass_index'] for e in complete]==[1,2,3,4,5]
    assert all(e['completed_documents']==stream['documents'] for e in complete)
    assert len({(e['process_instance'],e['pid']) for e in observations})==1
    assert [e['sequence'] for e in observations]==list(range(1,len(observations)+1))


def test_failing_final_root_never_emits_success(prepared,observations):
    directory,_=prepared;stream=read_json(directory/'wikipedia/stream.json');stream['index_root']='f'*64
    with pytest.raises(EvidenceError):m.validate_stream(directory/'wikipedia',stream)
    assert observations[-1]['completed_documents']==stream['documents']
    assert not any(e['complete'] for e in observations)


def test_failing_cursor_census_keeps_inner_validation_only(prepared,observations):
    directory,manifest=prepared;directory=directory/'wikipedia';stream=read_json(directory/'stream.json');r=recipe(manifest['tokenizer']['vocab_size'])
    total=coverage.schedule_counts(directory,r)['updates']
    with pytest.raises(EvidenceError,match='exact full-stream final'):m.boundary_cursors(directory,r,digest(stream),[0,total+1])
    assert [e['operation'] for e in observations if e['complete']]==['stream-validation']
    assert observations[-1]['operation']=='boundary-cursor-census' and not observations[-1]['complete']


def test_pass_ceiling_is_finite_and_failure_releases_lock(prepared,observations,monkeypatch):
    directory,_=prepared;directory=directory/'wikipedia';stream=read_json(directory/'stream.json')
    monkeypatch.setattr(m,'_pass_index',m.MAX_PASSES-1)
    m.validate_stream(directory,stream)
    before=len(observations)
    with pytest.raises(EvidenceError,match='pass limit'):m.validate_stream(directory,stream)
    assert len(observations)==before
    # The test explicitly resets process-local state; production has no reset API.
    monkeypatch.setattr(m,'_pass_index',0);m.validate_stream(directory,stream)


def test_concurrent_scan_cannot_share_the_observer(prepared,observations):
    directory,_=prepared;directory=directory/'wikipedia';stream=read_json(directory/'stream.json');errors=[]
    def other_thread():
        try:m.validate_stream(directory,stream)
        except Exception as error:errors.append(error)
    with m._lock:
        thread=threading.Thread(target=other_thread);thread.start();thread.join(5)
        assert not thread.is_alive()
    assert len(errors)==1 and isinstance(errors[0],EvidenceError) and not observations
    m.validate_stream(directory,stream)


def test_disabled_census_calls_original_directly_without_extra_manifest_read(tmp_path,monkeypatch):
    monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False);seen=[]
    monkeypatch.setattr(coverage,'schedule_counts',lambda *a:seen.append(a) or 42)
    monkeypatch.setattr(production_cursors,'boundary_cursors',lambda *a:seen.append(a) or 43)
    assert m.schedule_counts(tmp_path,{})==42
    assert m.boundary_cursors(tmp_path,{},'a'*64,[0,1])==43
    assert len(seen)==2


def test_nested_row_iterator_is_rejected_without_mutating_source_globals(prepared,observations,monkeypatch):
    directory,_=prepared;wiki=read_json(directory/'wikipedia/stream.json');chat=read_json(directory/'conversation/stream.json')
    original=data.rows
    def nested(path):
        with pytest.raises(EvidenceError,match='nested production row'):
            m.validate_stream(directory/'conversation',chat)
        yield from original(path)
    monkeypatch.setattr(data,'rows',nested)
    assert m.validate_stream(directory/'wikipedia',wiki)==wiki['targets']
    assert data.rows is nested and {e['stream_sha256'] for e in observations}=={digest(wiki)}


def test_actual_complete_cpu_replay_rehashes_reused_validation_and_keeps_all_state_checks(cpu_runtime,prepared,tmp_path,monkeypatch,observations):
    from test_production_replay import setup
    registration,envelopes,key,prover,run=setup(prepared,tmp_path,monkeypatch)
    result=run(tmp_path/'observed-verifier')
    assert result['result']=='PASS' and len(result['comparisons'])==len(envelopes)
    assert result['targets_recomputed']=={p:c['targets'] for p,c in registration['coverage'].items()}
    assert result['initial_state_regenerated'] is True and result['prover_checkpoints_restored'] is False
    final=[e for e in observations if e['complete']]
    assert [e['pass_index'] for e in final]==list(range(1,7))
    assert [e['operation'] for e in final].count('stream-validation')==2
    assert [e['operation'] for e in final].count('coverage-census')==2
    assert [e['operation'] for e in final].count('boundary-cursor-census')==2


def test_cpu_record_recovery_revalidates_with_fresh_scope_under_explicit_observer_reset(cpu_runtime,prepared,tmp_path,monkeypatch,observations):
    from test_production_record import setup
    from ovl_pipeline import production_record as recorder
    registration,expected,key,streams,run=setup(prepared,tmp_path,monkeypatch)
    stopped=False
    def anchor(r,root,envelopes,*a,**kw):
        nonlocal stopped
        boundary=envelopes[-1]['body']
        if not stopped and boundary['kind']=='progress':
            stopped=True;raise EvidenceError('explicit interrupted publisher substitute')
        assert boundary['control']==expected[boundary['index']]['body']['control']
        assert boundary['checkpoint']['state_root']==expected[boundary['index']]['body']['checkpoint']['state_root']
        return {'explicit-test-double':True}
    monkeypatch.setattr(recorder,'await_anchor',anchor)
    with pytest.raises(EvidenceError,match='interrupted publisher'):run()
    assert len([e for e in observations if e['complete']])==4
    # This is a same-process CPU test with explicit observer reset, not evidence
    # that a new OS process or actual public endorsement ran.
    monkeypatch.setattr(m,'_pass_index',0)
    for k,v in (('_last',None),('_sequence',0),('_process',None)):monkeypatch.setattr(runtime_activity,k,v)
    observations.clear()
    result=run(resume=True)
    assert result['result']=='RECORDED_NOT_REPLAYED' and result['recording_resumed'] is True
    assert [e['pass_index'] for e in observations if e['complete']]==list(range(1,9))
