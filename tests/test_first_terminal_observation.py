"""Adversarial peer terminal changes across shutdown observations."""
import copy
import pytest
import sustained_pilot_abort as abort
from ovl_pipeline.canonical import EvidenceError,read_json,write_json
from ovl_pipeline.supervision import Journal
from workload_health import Health
from test_workload_stage import staged,intent
from test_failed_stage_retention import failed,arguments


def test_first_preliminary_terminal_remains_binding_across_second_observation(tmp_path,monkeypatch):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        original=abort.job_supervision;observations=[]
        def changed(*a,**k):
            value=original(*a,**k)
            if observations:
                terminal=remote/'jobs'/root/'exit.json';new=read_json(terminal);new['exit_code']=17;write_json(terminal,new)
                value=copy.deepcopy(value);value['terminal']=new
            observations.append(value);return value
        monkeypatch.setattr(abort,'job_supervision',changed)
        with pytest.raises(EvidenceError):abort.stop_and_retain(*args,**kw)
        assert not h.jobs[root]['finished'] and not(out/'failure-retention/retention.json').exists()
        before=len(calls)
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):abort.stop_and_retain(*args,**kw)
        assert len(calls)==before


def test_abandonment_response_cannot_be_replaced_by_later_terminal(tmp_path,monkeypatch):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        first={'schema':'ovl.workload-job-abandonment.v1','job_sha256':root,'state':'ABANDONED',
               'observed_child':None,'exit_code':'UNAVAILABLE','scope':'synthetic initial terminal'}
        second={**first,'scope':'synthetic contradictory terminal'};abandoned=[]
        def observe(*a,**k):
            if k.get('abandon'):
                abandoned.append(True);return first
            if abandoned:return {'state':'ABANDONED','terminal':second}
            return {'state':'SUPERVISOR_ABSENT'}
        monkeypatch.setattr(abort,'job_supervision',observe)
        with pytest.raises(EvidenceError):abort.stop_and_retain(*args,**kw)
        assert read_json(out/'terminal-observation.json')==first
        before=len(calls)
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):abort.stop_and_retain(*args,**kw)
        assert len(calls)==before and not h.jobs[root]['finished']


def test_previously_pinned_normal_stage_terminal_remains_binding_in_backup(tmp_path):
    from run_workload_stage import retain_terminal
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        terminal=remote/'jobs'/root/'exit.json';first=read_json(terminal);retain_terminal(out,root,first)
        write_json(terminal,{**first,'exit_code':17})
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        with pytest.raises(EvidenceError):abort.stop_and_retain(*args,**kw)
        assert read_json(out/'terminal-observation.json')==first and not h.jobs[root]['finished']
