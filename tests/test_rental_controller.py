"""Deterministic provider/health integration only; never allocates paid resources."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json,hashlib
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_rental_controller as controller
from test_external_watchdog import intent as watchdog_intent,NOW,Fake
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal,rental_plan
from rental_quote import AUTHORIZATION_SHA256


def intent():
    w=watchdog_intent();w['baseline']['balance_usd']='100'
    payload={**w['payload'],'cloudType':'SECURE','gpuTypeId':'NVIDIA RTX 5090','minVcpuCount':1,'minMemoryInGb':1,
             'dockerArgs':'','startSsh':True,'startJupyter':False,'ports':'22/tcp'}
    quote={'schema':'ovl.rental-quote.v1','observed_epoch':NOW,'catalog_response_sha256':'c'*64,
           'selected_gpu':{'id':'NVIDIA RTX 5090','secure':True,'secure_hourly_usd':'0.24'},
           'storage_source':'https://docs.runpod.io/pods/pricing','storage_page_sha256':'d'*64,'storage_observed_epoch':NOW,
           'container_gb_month_usd':'0.10','volume_gb_month_upper_usd':'0.20','monthly_hours':672,'rate_margin_percent':125}
    quote['catalog_response']=json.dumps({'gpus':[{'id':'NVIDIA RTX 5090','secure':True,'maxCount':{'secure':1},'price':{'secure':'0.24'}}]})
    quote['catalog_response_sha256']=hashlib.sha256(quote['catalog_response'].encode()).hexdigest()
    v={**w['plan']['input'],'quote_sha256':digest(quote),'authorization_sha256':AUTHORIZATION_SHA256,'hourly_upper_usd':'0.300745'}
    w['plan']=rental_plan(v)
    return {'schema':'ovl.rental-controller-intent.v1','watchdog_intent':w,'payload':payload,'quote':quote}

class RentalFake(Fake):
    def __init__(self,path):
        self.value=intent();super().__init__(self.value['watchdog_intent']);self.alive=False
        self.heartbeat=path/'watchdog.json';self.health=path/'workload.json';self.directory=path/'controller'
        self.balance='100';self.refresh_heartbeat=True;self.healthy=True;self.completed=False
        self.create_failure=False;self.create_crash=False;self.writes=0;self.before_create=None
        self.refresh()
    def refresh(self):
        if self.refresh_heartbeat:
            write_json(self.heartbeat,{'schema':'ovl.external-watchdog-heartbeat.v1','intent_sha256':digest(self.i),
                'plan_sha256':digest(self.i['plan']),'external_terminate_epoch':self.i['plan']['external_terminate_epoch'],
                'observed_epoch':int(self.now),'state':'ARMED','pod_id':'owned-pod' if self.alive else None})
        if self.healthy and self.alive:
            write_json(self.health,{'schema':'ovl.rental-workload-health.v1','intent_sha256':digest(self.i),'pod_id':'owned-pod',
                'observed_epoch':int(self.now),'progress_epoch':int(self.now),'exported_checkpoint_epoch':int(self.now),'complete':self.completed})
    def account(self):
        self.refresh();return {**super().account(),'balance_usd':self.balance}
    def provider(self,operation,variables=None):
        if operation!='create':return super().request(operation,variables)
        self.calls.append((operation,variables,self.now));self.writes+=1
        event=read_json(self.directory/'event-00000000.json');assert event['body']==self.value and event['kind']=='creation-intent'
        assert variables=={'input':self.value['payload']}
        if self.before_create:self.before_create()
        self.alive=True
        if self.create_crash:raise KeyboardInterrupt('simulated process death after server creation')
        if self.create_failure:raise TimeoutError('simulated uncertain create response')
        return {'podFindAndDeployOnDemand':self.pod()},'a'*64,{}
    def run(self):
        controller.run(self.directory,self.value,digest(self.value),self.heartbeat,self.health,
            get_account=self.account,provider_request=self.provider,wall=lambda:self.now,monotonic=lambda:self.elapsed,sleep=self.sleep,
            boot=lambda:{'boot_id':'fake-boot','boottime_ms':int(self.elapsed*1000)},fence_root=self.directory.parent/'fences')


def test_one_shot_create_graceful_shutdown_and_verified_absence(tmp_path):
    f=RentalFake(tmp_path);f.run()
    assert f.writes==1 and not f.alive
    stop=read_json(f.directory/'stop-request.json')
    assert stop['observed_epoch']==f.i['plan']['request_checkpoint_epoch']
    first=next(c for c in f.calls if c[0]=='terminate')
    assert first[2]==f.i['plan']['provider_terminate_epoch']
    result=read_json(f.directory/'result.json')
    assert result['confirmed_absent_epoch']>=first[2]+15 and result['provider_billing_reconciliation']=='PENDING'
    f.run();assert f.writes==1


def test_unknown_create_response_is_adopted_and_aborted_never_reissued(tmp_path):
    f=RentalFake(tmp_path);f.create_failure=True;f.run()
    assert f.writes==1 and not f.alive
    assert any(c[0]=='identities' for c in f.calls)
    f.run();assert f.writes==1


def test_crash_after_remote_creation_resumes_by_identity_only(tmp_path):
    f=RentalFake(tmp_path);f.create_crash=True
    with pytest.raises(KeyboardInterrupt):f.run()
    assert f.alive and f.writes==1
    f.create_crash=False;f.run()
    assert not f.alive and f.writes==1
    with Journal(f.directory).lease() as j:
        assert any(e['body'].get('adopted_from_unique_intent') for e in j.events)


def test_intent_durable_but_no_request_never_recreates(tmp_path):
    f=RentalFake(tmp_path)
    with Journal(f.directory).lease() as j:j.append('creation-intent',f.value)
    f.run();assert f.writes==0 and read_json(f.directory/'result.json')['pod_id'] is None


@pytest.mark.parametrize('change',['missing','stale','wrong-root','terminating'])
def test_creation_requires_fresh_exact_armed_watchdog(tmp_path,change):
    f=RentalFake(tmp_path);h=read_json(f.heartbeat)
    if change=='missing':f.heartbeat.unlink()
    else:
        if change=='stale':h['observed_epoch']-=31
        if change=='wrong-root':h['intent_sha256']='d'*64
        if change=='terminating':h['state']='TERMINATING'
        write_json(f.heartbeat,h)
    with pytest.raises((EvidenceError,FileNotFoundError)):f.run()
    assert f.calls==[]


def test_watchdog_death_after_create_terminates_early(tmp_path):
    f=RentalFake(tmp_path);f.refresh_heartbeat=False;f.run()
    assert next(c for c in f.calls if c[0]=='terminate')[2]==NOW+40
    assert not f.alive


def test_no_workload_health_cannot_leave_idle_pod_indefinitely(tmp_path):
    f=RentalFake(tmp_path);f.healthy=False;f.run()
    # Fixed checkpoint deadline also bounds setup; no heartbeat grants a renewal.
    assert next(c for c in f.calls if c[0]=='terminate')[2]<=NOW+605


def test_completed_export_stops_without_waiting_for_deadline(tmp_path):
    f=RentalFake(tmp_path);f.completed=True;f.run()
    assert next(c for c in f.calls if c[0]=='terminate')[2]==NOW


def test_debit_exceeding_reserved_rental_ceiling_aborts(tmp_path):
    f=RentalFake(tmp_path);f.before_create=lambda:setattr(f,'balance','98');f.run()
    assert next(c for c in f.calls if c[0]=='terminate')[2]==NOW


def test_spend_uses_upper_rate_accrual_and_retains_prior_reservations(tmp_path):
    f=RentalFake(tmp_path);f.alive=True;f.now+=100;obs=f.account();h=read_json(f.heartbeat)
    v=controller.normalized(f.i,obs,f.pod(),h,read_json(f.health),f.now)
    assert Decimal(v['outstanding_usd'])>=Decimal('.15')+Decimal(100)*Decimal('.3')/3600
    assert v['reserved_remaining_usd']=='60'
    obs['balance_usd']='99.5';v=controller.normalized(f.i,obs,f.pod(),h,read_json(f.health),f.now)
    assert v['outstanding_usd']=='0.500000'


def test_changed_singleton_account_is_not_modified_before_create(tmp_path):
    f=RentalFake(tmp_path);f.alive=True;f.refresh()
    with pytest.raises(EvidenceError):f.run()
    assert f.calls==[] and f.alive


@pytest.mark.parametrize('field,value',[('gpuCount',2),('terminateAfter','2099-01-01T00:00:00Z'),('startJupyter',True),('ports','8888/http'),('dockerArgs','run arbitrary'),('env',[])])
def test_changed_or_secret_bearing_payload_refused(tmp_path,field,value):
    f=RentalFake(tmp_path);f.value['payload'][field]=value
    with pytest.raises(EvidenceError):f.run()
    assert f.calls==[]


def test_duplicate_service_leaves_live_controller_alone(tmp_path):
    f=RentalFake(tmp_path)
    with Journal(f.directory).lease():
        with pytest.raises(EvidenceError,match='another controller'):f.run()
    assert f.calls==[]


def test_observed_balance_rounds_down_and_costs_round_up():
    assert controller.dollars(Decimal('0.1234561'))=='0.123457'
    assert controller.dollars(Decimal('0.1234569'),balance=True)=='0.123456'


def test_completed_work_requires_final_export(tmp_path):
    f=RentalFake(tmp_path);f.alive=True;f.completed=True;f.refresh();h=read_json(f.health)
    h['exported_checkpoint_epoch']-=1
    with pytest.raises(EvidenceError,match='final state exported'):
        controller.normalized(f.i,f.account(),f.pod(),read_json(f.heartbeat),h,NOW)


def test_prior_stop_cannot_be_cancelled_by_healthy_restart(tmp_path):
    f=RentalFake(tmp_path);f.alive=True;f.now=NOW+100;f.refresh()
    with Journal(f.directory).lease() as j:
        j.append('creation-intent',f.value);j.append('creation-observed',{'id':'owned-pod'})
        controller.Lifetime(j,f.i['plan'],wall=lambda:NOW,clock=lambda:{'boot_id':'fake-boot','boottime_ms':0},initialize=True)
        j.append('decision',{'action':'CHECKPOINT_AND_STOP','observed_epoch':NOW})
    f.run();assert f.writes==0
    assert next(c for c in f.calls if c[0]=='terminate')[2]==NOW+300


def test_absence_between_liveness_reads_does_not_prove_teardown(tmp_path):
    f=RentalFake(tmp_path);original=f.account
    def flicker():
        obs=original()
        if f.alive and f.reads in (3,4):obs['pods']=[]
        return obs
    f.account=flicker;f.run()
    assert not f.alive
    assert read_json(f.directory/'result.json')['confirmed_absent_epoch']>=f.i['plan']['provider_terminate_epoch']+15
