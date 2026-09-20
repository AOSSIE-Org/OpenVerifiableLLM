"""Actual local stage processes across phase completion on one retained clock."""
from pathlib import Path
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_sustained_pilot as m
from production_run_health import ProductionRunHealth
from test_sustained_pilot_dispatch import fixture
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal


def phase(plan,index,prior):
    return {**plan,'schema':'ovl.sustained-pilot-plan.v2','stages':[plan['stages'][index]],
            'uploads':plan['uploads'] if index==0 else [],'prior_jobs':sorted(prior),
            'completion_scope':'phase-only-under-production-run-coordinator'}


def configured(tmp_path):
    _,plan,rental,t,remote,calls,controller,heartbeat=fixture(tmp_path,second=True)
    def health(j):return ProductionRunHealth(j,rental['watchdog_intent'],t.profile['pod_id'],None,{}, {})
    def run(p,h,name):return m.run(p,digest(p),rental,controller,heartbeat,t,tmp_path,
        Path(__file__).parents[1]/'scripts/pod_job_worker.py',tmp_path/name,tmp_path/'health.json',
        run_health=h,sleep=lambda _:time.sleep(.01))
    return plan,rental,t,remote,calls,health,run


def test_two_phases_reuse_clock_export_age_and_launch_fences_without_finishing_rental(tmp_path):
    plan,rental,t,remote,calls,health,run=configured(tmp_path);journal=tmp_path/'run-journal'
    with Journal(journal).lease() as j:
        h=health(j);one=phase(plan,0,[]);first=run(one,h,'phase-one')
        original_clock=[e for e in j.events if e['body'].get('action')=='LIFETIME_CLOCK']
        assert len(original_clock)==1 and not read_json(tmp_path/'health.json')['complete']
        before=len(calls);assert run(one,h,'phase-one')==first and len(calls)==before
        prior=list(h.jobs);exported=h.exported
    with Journal(journal).lease() as j:
        h=health(j);assert h.exported==exported and not h.complete
        two=phase(plan,1,prior);second=run(two,h,'phase-two')
        assert second['outcome']=='EXITED_ZERO' and len(h.jobs)==2 and not h.complete
        assert [e for e in j.events if e['body'].get('action')=='LIFETIME_CLOCK']==original_clock
        assert not read_json(tmp_path/'health.json')['complete']
        before=len(calls);assert run(two,h,'phase-two')==second and len(calls)==before
        assert len([c for c in calls if ' start ' in c[-1]])==2


@pytest.mark.parametrize('damage',['omitted-prior','unknown-prior','changed-prior-descriptor','overlap',
                                  'unleased','wrong-pod','wrong-rental','missing-run-health','historical-schema'])
def test_changed_phase_admission_fails_before_new_remote_work(tmp_path,damage):
    plan,rental,t,remote,calls,health,run=configured(tmp_path);journal=tmp_path/'run-journal'
    with Journal(journal).lease() as j:
        h=health(j);run(phase(plan,0,[]),h,'phase-one');two=phase(plan,1,list(h.jobs));before=len(calls)
        if damage=='omitted-prior':two['prior_jobs']=[]
        elif damage=='unknown-prior':two['prior_jobs']=['f'*64]
        elif damage=='changed-prior-descriptor':
            result=read_json(tmp_path/'phase-one/stages/first/stage-result.json')
            p=Path(result['exports'][0]['directory'])/'job.json';p.chmod(0o600);p.write_bytes(b'{}')
        elif damage=='overlap':
            p=tmp_path/two['stages'][0]['template_path'];v=read_json(p)
            old=read_json(tmp_path/plan['stages'][0]['template_path']);v['export_roots']=old['export_roots']
            write_json(p,v);two['stages'][0]['template_sha256']=digest(v)
        elif damage=='wrong-pod':h.pod='another-pod'
        elif damage=='wrong-rental':h.root='e'*64
        elif damage=='missing-run-health':h=None
        elif damage=='historical-schema':
            two['schema']='ovl.sustained-pilot-plan.v1';del two['prior_jobs'];del two['completion_scope']
        fd=j._fd
        if damage=='unleased':j._fd=None
        try:
            with pytest.raises(EvidenceError):run(two,h,'phase-two')
        finally:j._fd=fd
        assert len(calls)==before


def test_phase_refuses_when_complete_remaining_phase_budget_does_not_fit(tmp_path):
    plan,rental,t,remote,calls,health,run=configured(tmp_path)
    with Journal(tmp_path/'run-journal').lease() as j:
        h=health(j);one=phase(plan,0,[])
        # The original deadline is left untouched; the selected phase is late.
        h.now=lambda:rental['watchdog_intent']['plan']['request_checkpoint_epoch']-10
        with pytest.raises(EvidenceError,match='remaining original work window'):run(one,h,'phase-one')
        assert not calls and not h.complete
