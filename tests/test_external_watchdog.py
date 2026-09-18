"""External controller tests use a deterministic fake provider; no rental credit."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_external_watchdog as watchdog
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import Journal,rental_plan

NOW=1800000000

def intent():
    plan=rental_plan({'schema':'ovl.rental-budget-input.v2','attempt_id':'ovllm-test-12345678',
        'now_epoch':NOW,'spent_usd':'0.068185','outstanding_usd':'0.15','reserved_remaining_usd':'60',
        'allowance_usd':'1','hourly_upper_usd':'0.3','quote_sha256':'a'*64,'maximum_seconds':600,
        'checkpoint_grace_seconds':300,'billing_slack_seconds':300,
        'external_termination_grace_seconds':120,'authorization_sha256':'b'*64})
    return {'schema':'ovl.external-watchdog-intent.v1','plan':plan,'creation_latest_epoch':NOW+30,
            'payload':{'name':plan['input']['attempt_id'],'gpuCount':1,'imageName':'ubuntu@sha256:'+'c'*64,
                       'containerDiskInGb':4,'volumeInGb':0,
                       'terminateAfter':datetime.fromtimestamp(plan['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},
            'baseline':{'observed_epoch':NOW,'pods':[],'volume_ids':[],'autopay':False,'account_hourly_usd':'0',
                        'http_clock':{'server_epoch':NOW,'request_started_epoch':NOW,'request_completed_epoch':NOW}}}

class Fake:
    def __init__(self,i):
        self.i=i;self.now=NOW;self.elapsed=0;self.alive=True;self.calls=[];self.reads=0
        self.fail_reads=False;self.fail_terminate=0;self.hide_first=False;self.shape_change={}
    def sleep(self,seconds):
        self.now+=seconds;self.elapsed+=seconds
        assert self.elapsed<2000,'watchdog failed to finish bounded fake test'
    def pod(self):
        return {**{k:v for k,v in self.i['payload'].items() if k!='terminateAfter'},
                'id':'owned-pod','createdAt':datetime.fromtimestamp(NOW,timezone.utc).isoformat(),
                'costPerHr':'0.24','adjustedCostPerHr':'0.24',**self.shape_change}
    def account(self):
        self.reads+=1
        if self.fail_reads:raise TimeoutError('fake account outage')
        return {'observed_epoch':int(self.now),'pods':[self.pod()] if self.alive and not(self.hide_first and self.reads==1) else [],
                'volume_ids':[],'autopay':False,'account_hourly_usd':'0.24' if self.alive else '0',
                'http_clock':{'server_epoch':int(self.now),'request_started_epoch':int(self.now),'request_completed_epoch':int(self.now)}}
    def request(self,operation,variables=None):
        self.calls.append((operation,variables,self.now))
        assert operation in ('identities','terminate'),'watchdog must never create'
        if operation=='identities':return {'myself':{'pods':[self.pod()] if self.alive else []}},'a'*64,{}
        assert variables=={'input':{'podId':'owned-pod'}}
        if self.fail_terminate:
            self.fail_terminate-=1;raise TimeoutError('fake termination timeout')
        self.alive=False;self.fail_reads=False
        return {},'b'*64,{}
    def run(self,path):
        watchdog.run(path,self.i,digest(self.i),get_account=self.account,provider_request=self.request,
                     wall=lambda:self.now,monotonic=lambda:self.elapsed,sleep=self.sleep)


def test_enforces_grace_deadline_and_two_fresh_absence_observations(tmp_path):
    i=intent();f=Fake(i);f.run(tmp_path/'journal')
    term=[x for x in f.calls if x[0]=='terminate']
    assert term[0][2]==i['plan']['external_terminate_epoch']
    r=read_json(tmp_path/'journal/result.json')
    assert r['complete'] and r['confirmed_absent_epoch']>=term[0][2]+15
    assert r['automatic_provider_termination']=='UNVERIFIED'
    assert r['provider_billing_reconciliation']=='PENDING'
    before=list(f.calls);f.run(tmp_path/'journal');assert f.calls==before


def test_pre_creation_absence_does_not_complete_guard(tmp_path):
    i=intent();f=Fake(i);f.hide_first=True;f.run(tmp_path/'journal')
    assert any(x[0]=='terminate' for x in f.calls)
    assert read_json(tmp_path/'journal/result.json')['pod_id']=='owned-pod'


def test_account_failure_recovers_identity_and_terminates_without_waiting(tmp_path):
    i=intent();f=Fake(i);f.fail_reads=True;f.fail_terminate=2;f.run(tmp_path/'journal')
    assert f.calls[0][0]=='identities'
    term=[x for x in f.calls if x[0]=='terminate']
    assert len(term)>=3 and term[0][2]==NOW
    assert read_json(tmp_path/'journal/result.json')['complete']


def test_adopts_known_id_and_terminates_before_deadline_read(tmp_path):
    i=intent();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        j.append('creation-intent',i);j.append('creation-observed',{'id':'owned-pod'})
    f=Fake(i);f.now=i['plan']['external_terminate_epoch'];f.fail_reads=True;f.run(path)
    assert f.calls[0][0]=='terminate' and f.calls[0][2]==f.i['plan']['external_terminate_epoch']


def test_monotonic_limit_survives_backward_wall_clock(tmp_path):
    i=intent();f=Fake(i);original=f.sleep
    def rollback(seconds):
        original(seconds)
        if f.elapsed==100:f.now-=50
    f.sleep=rollback;f.run(tmp_path/'journal')
    first=next(x for x in f.calls if x[0]=='terminate')
    assert first[2]<=i['plan']['external_terminate_epoch']


@pytest.mark.parametrize('change',[{'gpuCount':2},{'imageName':'wrong'},{'costPerHr':'0.31'}])
def test_misprovisioned_attributed_pod_is_terminated(tmp_path,change):
    i=intent();f=Fake(i);f.shape_change=change;f.run(tmp_path/'journal')
    assert next(x for x in f.calls if x[0]=='terminate')[2]==NOW


def test_refuses_modified_deadline_before_any_provider_calls(tmp_path):
    i=intent();f=Fake(i);pin=digest(i);i['payload']['terminateAfter']='2099-01-01T00:00:00Z'
    with pytest.raises(EvidenceError):watchdog.validate_intent(i,pin)
    with pytest.raises(EvidenceError):f.run(tmp_path/'journal')
    assert f.calls==[] and f.reads==0


def test_corrupt_journal_triggers_attributed_teardown_without_deleting_evidence(tmp_path):
    i=intent();f=Fake(i);path=tmp_path/'journal';path.mkdir();event=path/'event-00000000.json';event.write_bytes(b'corrupted')
    with pytest.raises(EvidenceError):
        watchdog.run_guarded(path,i,digest(i),get_account=f.account,provider_request=f.request,
            wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep)
    assert event.read_bytes()==b'corrupted' and not f.alive
    assert not (path/'result.json').exists()


def test_duplicate_controller_never_terminates_live_owner_resource(tmp_path):
    i=intent();f=Fake(i);path=tmp_path/'journal'
    with Journal(path).lease():
        with pytest.raises(EvidenceError,match='another controller'):
            watchdog.run_guarded(path,i,digest(i),get_account=f.account,provider_request=f.request,
                wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep)
    assert f.calls==[] and f.alive
