import copy
from pathlib import Path
import subprocess
import sys

import pytest

from ovl_pipeline.canonical import EvidenceError, canonical, digest
from ovl_pipeline.supervision import Journal, observe, rental_plan


def plan_input():
    return {"schema": "ovl.rental-budget-input.v1", "attempt_id": "pilot-v1", "now_epoch": 1800000000,
            "spent_usd": "0", "outstanding_usd": "0", "reserved_remaining_usd": "60",
            "allowance_usd": "10", "hourly_upper_usd": "1.2", "quote_sha256": "a" * 64,
            "maximum_seconds": 3600, "checkpoint_grace_seconds": 600, "billing_slack_seconds": 300}


def observation(plan):
    now = plan["input"]["now_epoch"] + 100
    return {"schema": "ovl.supervisor-observation.v1", "now_epoch": now, "observed_epoch": now,
            "attributed_pod_ids": ["owned-pod"], "active_pod_ids": ["owned-pod", "unrelated"],
            "pod_id": "owned-pod", "gpu_count": 1, "hourly_usd": "1.2",
            "provider_terminate_epoch": plan["provider_terminate_epoch"], "provider_guard_verified": True,
            "actual_project_spend_usd": "0.1", "outstanding_usd": "0", "reserved_remaining_usd": "60",
            "account_balance_usd": "99.9", "progress_epoch": now, "last_checkpoint_epoch": now}


def test_absolute_deadline_charges_setup_billing_slack_and_reserves():
    p = rental_plan(plan_input())
    assert p["maximum_charge_micro_usd"] == 1_300_000
    assert p["provider_terminate_epoch"] == 1800003600
    assert p["request_checkpoint_epoch"] == 1800003000
    assert p["billing_ceiling_epoch"] == 1800003900
    assert p["execution_admission"] == "NOT_RUN" and p["provider_guard"] == "NOT_RUN"
    v = plan_input();v.update(spent_usd="29", maximum_seconds=100000)
    p = rental_plan(v)
    assert p["maximum_charge_micro_usd"] == 1_000_000
    assert p["provider_terminate_epoch"] == 1800002700
    v["spent_usd"] = "29.999"
    with pytest.raises(EvidenceError, match="remaining funds"):rental_plan(v)


def test_continue_is_policy_only_and_unrelated_pods_are_not_targeted():
    p = rental_plan(plan_input());v = observation(p);r = observe(p, v)
    assert r["action"] == "CONTINUE" and r["provider_mutation"] == "NOT_RUN"
    assert r["managed_active_pod_ids"] == ["owned-pod"] and r["unrelated_active_pod_ids"] == ["unrelated"]
    v["pod_id"] = "unrelated"
    with pytest.raises(EvidenceError, match="attribution"):observe(p, v)


@pytest.mark.parametrize("changes,reason", [
    ({"gpu_count": 2}, "project-singleton-or-device-count"),
    ({"attributed_pod_ids": ["owned-pod", "unrelated"]}, "project-singleton-or-device-count"),
    ({"provider_guard_verified": False}, "missing-or-changed-provider-deadline"),
    ({"provider_terminate_epoch": 1800003601}, "missing-or-changed-provider-deadline"),
    ({"observed_epoch": 1800000000}, "stale-or-inconsistent-provider-observation"),
    ({"observed_epoch": 1800000101}, "stale-or-inconsistent-provider-observation"),
    ({"progress_epoch": 1799999700}, "stalled-or-future-progress"),
    ({"last_checkpoint_epoch": 1799998000}, "missing-recent-durable-checkpoint"),
    ({"hourly_usd": "1.200001"}, "quote-upper-bound-exceeded"),
    ({"reserved_remaining_usd": "59.99"}, "reservation-regressed"),
    ({"actual_project_spend_usd": "30"}, "operating-budget-guard"),
    ({"account_balance_usd": "70"}, "account-balance-insufficient-for-reserved-work"),
    ({"now_epoch": 1800003000}, "checkpoint-deadline"),
])
def test_failed_guards_request_checkpoint_and_stop(changes, reason):
    p = rental_plan(plan_input());v = {**observation(p), **changes};r = observe(p, v)
    assert r["action"] == "CHECKPOINT_AND_STOP" and reason in r["reasons"]


def test_tampered_deadline_and_missing_fields_cannot_admit_work():
    p = rental_plan(plan_input());v = observation(p);p["provider_terminate_epoch"] += 3600
    with pytest.raises(EvidenceError):observe(p, v)
    p = rental_plan(plan_input());del v["provider_guard_verified"]
    with pytest.raises(EvidenceError):observe(p, v)


def test_journal_lease_is_exclusive_and_adopts_prior_intent(tmp_path):
    path = tmp_path / "controller";a = Journal(path);b = Journal(path)
    with pytest.raises(EvidenceError):a.append("creation-intent", {})
    with a.lease():
        root = a.append("creation-intent", {"attempt": "one", "result": "not-observed"})
        with pytest.raises(EvidenceError, match="another controller"):
            with b.lease():pass
    with b.lease():
        assert len(b.events) == 1 and b.events[0]["body"]["attempt"] == "one"
        b.append("creation-observed", {"resource_id": "pod1"})
        assert b.events[-1]["previous"] == root


@pytest.mark.parametrize("damage", ["truncate", "gap", "symlink", "parent"])
def test_journal_damage_fails_without_overwriting_evidence(tmp_path, damage):
    path = tmp_path / "controller"
    with Journal(path).lease() as journal:
        journal.append("creation-intent", {"attempt": "one"})
        journal.append("provider-observation", {"gpu_count": 0})
    first = path / "event-00000000.json";second = path / "event-00000001.json"
    if damage == "truncate":second.write_bytes(b'{"schema":')
    elif damage == "gap":second.rename(path / "event-00000003.json")
    elif damage == "symlink":second.unlink();second.symlink_to(first)
    else:
        import json
        event = json.loads(second.read_text());event["previous"] = "0" * 64;second.write_bytes(canonical(event))
    before = {p.name: p.read_bytes() for p in path.glob("event-*.json")}
    with pytest.raises(EvidenceError):
        with Journal(path).lease():pass
    assert before == {p.name: p.read_bytes() for p in path.glob("event-*.json")}


def test_process_death_releases_lease_and_preserves_fsynced_intent(tmp_path):
    path = tmp_path / "controller"
    code = "from pathlib import Path;from ovl_pipeline.supervision import Journal;import sys\nwith Journal(Path(sys.argv[1])).lease() as j:\n j.append('creation-intent',{'attempt':'crash-test'})\n print('ready',flush=True)\n sys.stdin.read()\n"
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(EvidenceError):
            with Journal(path).lease():pass
        child.kill();child.wait(timeout=10)
        with Journal(path).lease() as adopted:
            assert adopted.events[0]["body"] == {"attempt": "crash-test"}
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
        child.stdin.close();child.stdout.close()


def test_kill_before_atomic_publication_keeps_prior_intent_adoptable(tmp_path):
    path = tmp_path / "controller"
    with Journal(path).lease() as j:j.append("creation-intent", {"attempt":"already-created"})
    code = '''
from pathlib import Path
import sys
from ovl_pipeline import supervision
def stopped_link(*a,**k):
    print("pending-ready",flush=True)
    sys.stdin.read()
    raise AssertionError("test should kill this process")
with supervision.Journal(Path(sys.argv[1])).lease() as j:
    supervision.os.link=stopped_link
    j.append("creation-observed",{"pod_id":"owned"})
'''
    child=subprocess.Popen([sys.executable,"-c",code,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=="pending-ready"
        child.kill();child.wait(timeout=10)
        assert not (path/"event-00000001.json").exists()
        drafts=list(path.glob(".pending-*"));assert len(drafts)==1
        before=drafts[0].read_bytes()
        with Journal(path).lease() as j:
            assert len(j.events)==1 and j.events[0]["body"]["attempt"]=="already-created"
            j.append("creation-observed",{"pod_id":"owned"})
        assert drafts[0].read_bytes()==before  # Preserve uncommitted recovery evidence.
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
        child.stdin.close();child.stdout.close()


def revised_plan_input():
    return {**plan_input(),'schema':'ovl.rental-budget-input.v2',
            'external_termination_grace_seconds':120,'authorization_sha256':'b'*64}


def revised_observation(plan):
    value=observation(plan)
    del value['provider_guard_verified'];del value['provider_terminate_epoch']
    return {**value,'schema':'ovl.supervisor-observation.v2',
            'terminate_after_request_epoch':plan['provider_terminate_epoch'],
            'watchdog_observed_epoch':value['now_epoch'],'watchdog_plan_sha256':digest(plan),
            'watchdog_external_terminate_epoch':plan['external_terminate_epoch'],'watchdog_state':'ARMED'}


def test_revised_guard_charges_grace_separately_from_billing_slack():
    p=rental_plan(revised_plan_input())
    assert p['external_terminate_epoch']==p['provider_terminate_epoch']+120
    assert p['billing_ceiling_epoch']==p['external_terminate_epoch']+300
    assert p['maximum_charge_micro_usd']==1_340_000
    assert p['automatic_provider_termination']=='UNVERIFIED'
    assert p['external_watchdog']=='NOT_RUN'
    tight={**revised_plan_input(),'spent_usd':'29','maximum_seconds':100000}
    bounded=rental_plan(tight)
    assert bounded['provider_terminate_epoch']==1800002580  # 120s earlier than v1
    assert bounded['maximum_charge_micro_usd']==1_000_000
    assert observe(p,revised_observation(p))['action']=='CONTINUE'


@pytest.mark.parametrize('change',[
    {'external_termination_grace_seconds':119},{'external_termination_grace_seconds':121},
    {'external_termination_grace_seconds':True},{'authorization_sha256':'invalid'},
])
def test_revised_guard_requires_exact_authorized_grace_and_parent(change):
    with pytest.raises(EvidenceError):rental_plan({**revised_plan_input(),**change})


@pytest.mark.parametrize('change',[
    {'terminate_after_request_epoch':1800003601},{'watchdog_plan_sha256':'c'*64},
    {'watchdog_external_terminate_epoch':1800003721}, {'watchdog_observed_epoch':1800000069},
    {'watchdog_observed_epoch':1800000101},{'watchdog_state':'STOPPED'},
])
def test_revised_guard_refuses_stale_or_different_watchdog(change):
    p=rental_plan(revised_plan_input());v={**revised_observation(p),**change}
    result=observe(p,v)
    assert result['action']=='CHECKPOINT_AND_STOP'
    assert 'missing-stale-or-changed-external-watchdog' in result['reasons']


def test_revised_external_deadline_terminates_even_with_stale_observation():
    p=rental_plan(revised_plan_input());v=revised_observation(p)
    v['now_epoch']=p['external_terminate_epoch']
    r=observe(p,v)
    assert r['action']=='TERMINATE' and 'external-termination-deadline' in r['reasons']
    assert r['automatic_provider_termination']=='UNVERIFIED'
    assert r['provider_mutation']=='NOT_RUN'
