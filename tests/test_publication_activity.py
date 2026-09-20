"""Finite publisher transitions with explicit Actions responses; no public signing."""
from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import publication_activity as m
import publish_progress_boundary as publisher
from ovl_pipeline.canonical import EvidenceError,read_json


def test_repeated_stage_never_changes_bytes_and_cannot_renew_deadline(tmp_path):
    args=(tmp_path,'a'*64,'b'*64,'request-public-commit-verified',{'revision':'c'*40},1000)
    assert m.emit(*args,wall=lambda:100)
    path=tmp_path/'request-public-commit-verified.json';original=path.read_bytes();stamp=path.stat().st_mtime_ns
    assert not m.emit(*args,wall=lambda:999)
    assert path.read_bytes()==original and path.stat().st_mtime_ns==stamp
    with pytest.raises(EvidenceError,match='expired'):m.emit(*args,wall=lambda:1000)
    with pytest.raises(EvidenceError,match='changed'):m.emit(*args[:-1],1001,wall=lambda:100)
    value=read_json(path);assert m.validate(value,'a'*64,'b'*64,1000)==value
    value['identity']['revision']='d'*40
    with pytest.raises(EvidenceError,match='content differs'):m.validate(value,'a'*64,'b'*64,1000)


@pytest.mark.parametrize('stage',['poll','heartbeat','actions-step-00','actions-step-33',None])
def test_unbounded_or_nonmaterial_progress_stage_refused(tmp_path,stage):
    with pytest.raises(EvidenceError):m.emit(tmp_path,'a'*64,'b'*64,stage,{},1000,wall=lambda:100)
    assert not list(tmp_path.glob('*.json'))


def test_actions_polls_emit_only_distinct_identified_completed_steps(tmp_path):
    polls=[];new=[];clock=[100]
    def execute(args,**kwargs):
        if args[1:3]==['run','list']:
            polls.append(True)
            return json.dumps([{'databaseId':19,'headSha':'1'*40,'status':'completed' if len(polls)>1 else 'in_progress','conclusion':'success' if len(polls)>1 else None}])
        if args[1]=='api' and '/jobs?' in args[2]:
            return json.dumps({'total_count':1,'jobs':[{'id':42,'run_id':19,'head_sha':'1'*40,'name':'endorse-progress','steps':[
                {'number':1,'name':'Set up','status':'completed','conclusion':'success'},
                {'number':2,'name':'Verify','status':'completed' if len(polls)>1 else 'in_progress','conclusion':'success' if len(polls)>1 else None}]}]})
        if args[1]=='api':return json.dumps({'id':19,'head_sha':'1'*40,'head_branch':publisher.BRANCH,'path':publisher.PROGRESS_WORKFLOW,'event':'push','run_attempt':1,'status':'completed' if len(polls)>1 else 'in_progress','conclusion':'success' if len(polls)>1 else None})
        assert args[:3]==['gh','run','download'];Path(args[-1]).mkdir();return ''
    def progress(stage,identity):
        if m.emit(tmp_path/'activity','a'*64,'b'*64,stage,identity,200,wall=lambda:clock[0]):new.append(stage)
    def sleep(seconds):clock[0]+=seconds
    result=publisher.actions_artifact('1'*40,tmp_path/'artifact',200,execute=execute,wall=lambda:clock[0],sleep=sleep,progress=progress)
    assert result['run_id']==19 and len(polls)==2
    assert new==['actions-run-observed','actions-step-01','actions-step-02']
    assert len(list((tmp_path/'activity').glob('*.json')))==3


@pytest.mark.parametrize('damage',['job-head','job-run','extra-job','truncated-jobs','duplicate-step','too-many-steps','rerun'])
def test_changed_or_unbounded_actions_identity_cannot_report_step_progress(tmp_path,damage):
    stages=[]
    def execute(args,**kwargs):
        if args[1:3]==['run','list']:return json.dumps([{'databaseId':19,'headSha':'1'*40,'status':'in_progress','conclusion':None}])
        if '/jobs?' not in args[2]:
            return json.dumps({'id':19,'head_sha':'1'*40,'head_branch':publisher.BRANCH,'path':publisher.PROGRESS_WORKFLOW,'event':'push','run_attempt':2 if damage=='rerun' else 1})
        step={'number':1,'name':'Verify','status':'in_progress','conclusion':None}
        job={'id':42,'run_id':20 if damage=='job-run' else 19,'head_sha':'2'*40 if damage=='job-head' else '1'*40,'name':'endorse-progress','steps':[step]}
        if damage=='duplicate-step':job['steps']=[step,step]
        if damage=='too-many-steps':job['steps']=[step]*33
        jobs=[job,job] if damage=='extra-job' else [job]
        return json.dumps({'total_count':2 if damage=='truncated-jobs' else len(jobs),'jobs':jobs})
    with pytest.raises(EvidenceError):publisher.actions_artifact('1'*40,tmp_path/'artifact',200,execute=execute,wall=lambda:100,progress=lambda stage,identity:stages.append(stage))
    assert not any(s.startswith('actions-step-') for s in stages)
