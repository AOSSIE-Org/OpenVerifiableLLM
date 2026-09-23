"""Synthetic volume guards; no paid allocation or model acceptance credit."""
from copy import deepcopy
from decimal import Decimal
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_rental_controller as controller
import run_external_watchdog as watchdog
import retained_volume
from probe_provider_deadline import provision_errors
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import rental_plan
from test_rental_controller import intent,RentalFake
from test_external_watchdog import NOW,Fake


def selected():
    return {'schema':'ovl.retained-network-volume.v1','id':'synthetic-volume','name':'synthetic-cache',
            'size_gb':224,'data_center_id':'EU-RO-1','created_epoch':NOW-60,
            'retention_deadline_epoch':NOW-60+7*86400,'billing_ceiling_epoch':NOW-60+8*86400,
            'monthly_gb_usd':'0.07','monthly_hours':672,'reserved_usd':'4.480128'}


def observation(obs,selection):
    obs=deepcopy(obs);obs['volume_ids']=[selection['id']]
    obs['network_volumes']=[{'id':selection['id'],'name':selection['name'],'size':selection['size_gb'],'dataCenterId':selection['data_center_id']}]
    obs['account_hourly_usd']=str(Decimal(obs['account_hourly_usd'])+retained_volume.hourly(selection))
    for pod in obs['pods']:
        pod.update(networkVolumeId=selection['id'],networkVolumeDataCenterId=selection['data_center_id'],volumeMountPath='/workspace',volumeInGb=selection['size_gb'])
    return obs


def volume_intent():
    value=intent();w=value['watchdog_intent'];s=selected()
    value['schema']='ovl.rental-controller-intent.v6';w['schema']='ovl.external-watchdog-intent.v2';w['retained_volume']=s
    attachment={'networkVolumeId':s['id'],'dataCenterId':s['data_center_id'],'volumeMountPath':'/workspace','volumeInGb':s['size_gb']}
    w['payload'].update(attachment);value['payload'].update(attachment,allowedCudaVersions=['13.0'])
    w['baseline']=observation(w['baseline'],s)
    quote=value['quote'];quote.update(schema='ovl.rental-quote.v3',network_volume_sha256=digest(s))
    quote['selected_gpu']={'id':'NVIDIA RTX 5090','cloud':'SECURE','hourly_usd':'0.24'}
    w['plan']=rental_plan({**w['plan']['input'],'quote_sha256':digest(quote)})
    return value


def test_separate_storage_reservation_and_unchanged_deadline():
    v=volume_intent();w,p=controller.validate(v,digest(v))
    assert p['external_terminate_epoch']==intent()['watchdog_intent']['plan']['external_terminate_epoch']
    assert p['input']['hourly_upper_usd']=='0.300745'
    assert retained_volume.hourly(w['retained_volume'])*192==Decimal('4.480128')


@pytest.mark.parametrize('change',[
    {'id':'different'},{'size_gb':225},{'data_center_id':'EUR-NO-1'},
    {'reserved_usd':'4.479999'},{'monthly_gb_usd':'0.05'},{'monthly_hours':720},
    {'created_epoch':NOW+1},{'retention_deadline_epoch':NOW+10},
    {'retention_deadline_epoch':NOW+8*86400},{'billing_ceiling_epoch':NOW+9*86400},
])
def test_changed_identity_price_or_retention_fails_before_creation(change):
    v=volume_intent();v['watchdog_intent']['retained_volume'].update(change)
    with pytest.raises(EvidenceError):controller.validate(v,digest(v))


def test_insufficient_separate_storage_reservation_rejected():
    v=volume_intent();w=v['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'reserved_remaining_usd':'4.47'})
    with pytest.raises(EvidenceError,match='not reserved'):controller.validate(v,digest(v))


@pytest.mark.parametrize('field,value',[('networkVolumeId','another'),('dataCenterId','EUR-NO-1'),('volumeMountPath','/tmp'),('cloudType','COMMUNITY')])
def test_creation_attachment_and_cloud_cannot_diverge(field,value):
    v=volume_intent();v['payload'][field]=value
    with pytest.raises(EvidenceError):controller.validate(v,digest(v))


@pytest.mark.parametrize('change',['missing','extra','renamed','resized','moved','duplicate','missing-metadata'])
def test_unreviewed_observed_volumes_rejected(change):
    v=volume_intent();w=v['watchdog_intent'];obs=deepcopy(w['baseline'])
    if change=='missing':obs['volume_ids']=[];obs['network_volumes']=[]
    if change=='extra':obs['volume_ids'].append('other')
    if change=='duplicate':obs['volume_ids']*=2
    if change=='renamed':obs['network_volumes'][0]['name']='other'
    if change=='resized':obs['network_volumes'][0]['size']+=1
    if change=='moved':obs['network_volumes'][0]['dataCenterId']='EUR-NO-1'
    if change=='missing-metadata':del obs['network_volumes']
    assert not retained_volume.matches(w,obs)
    w['baseline']=obs
    with pytest.raises(EvidenceError):watchdog.validate_intent(w,digest(w))


class VolumeFake(RentalFake):
    def __init__(self,path):
        super().__init__(path);self.value=volume_intent();self.i=self.value['watchdog_intent'];self.refresh()
        self.volume_change=None
    def account(self):
        obs=observation(super().account(),self.i['retained_volume'])
        if self.alive and self.volume_change:self.volume_change(obs)
        return obs


def test_full_rental_cycle_retains_only_authorized_volume_and_never_deletes_it(tmp_path):
    f=VolumeFake(tmp_path);f.run()
    result=read_json(f.directory/'result.json')
    assert f.writes==1 and not f.alive and result['residual_network_volumes']==['synthetic-volume']
    assert result['provider_billing_reconciliation']=='PENDING'
    assert next(c for c in f.calls if c[0]=='terminate')[2]==f.i['plan']['provider_terminate_epoch']
    f.run();assert f.writes==1


@pytest.mark.parametrize('change',[
    lambda o:o['volume_ids'].append('unrelated'),
    lambda o:o['network_volumes'][0].update(size=225),
    lambda o:o['pods'][0].update(networkVolumeId='unrelated'),
    lambda o:o['pods'][0].update(networkVolumeDataCenterId='EUR-NO-1'),
    lambda o:o['pods'][0].update(volumeMountPath='/wrong'),
    lambda o:o.update(account_hourly_usd='0.5'),
])
def test_running_identity_or_rate_change_terminates_attributed_pod(tmp_path,change):
    f=VolumeFake(tmp_path);f.volume_change=change;f.run()
    assert not f.alive and f.writes==1
    assert next(c for c in f.calls if c[0]=='terminate')[2]<f.i['plan']['provider_terminate_epoch']


def test_small_storage_debit_before_creation_rechecks_funding(tmp_path):
    f=VolumeFake(tmp_path);f.balance='99.99';f.run();assert f.writes==1


def test_insufficient_funding_before_create_refuses_without_mutation(tmp_path):
    f=VolumeFake(tmp_path);f.balance='60'
    with pytest.raises(EvidenceError,match='account changed'):f.run()
    assert f.writes==0


def test_external_watchdog_preserves_deadline_and_volume(tmp_path):
    w=volume_intent()['watchdog_intent'];f=Fake(w);original=f.account
    f.account=lambda:observation(original(),w['retained_volume'])
    f.run(tmp_path/'watchdog')
    assert next(c for c in f.calls if c[0]=='terminate')[2]==w['plan']['external_terminate_epoch']
    assert read_json(tmp_path/'watchdog/result.json')['residual_network_volumes']==['synthetic-volume']


def test_old_intent_does_not_gain_storage_permission():
    v=intent();v['watchdog_intent']['retained_volume']=selected()
    with pytest.raises(EvidenceError):controller.validate(v,digest(v))


def test_excessive_debit_rejected_before_creation(tmp_path):
    f=VolumeFake(tmp_path);f.i['plan']=rental_plan({**f.i['plan']['input'],'spent_usd':'59'})
    f.refresh();f.balance='80'
    with pytest.raises(EvidenceError,match='pre-creation debit'):f.run()
    assert f.writes==0


def test_fresh_debit_must_leave_whole_rental_budget(tmp_path):
    f=VolumeFake(tmp_path)
    f.i['plan']=rental_plan({**f.i['plan']['input'],'spent_usd':'59.7'});f.refresh();f.balance='99.6'
    with pytest.raises(EvidenceError,match='operating budget'):f.run()
    assert f.writes==0


@pytest.mark.parametrize('guard',['controller','watchdog'])
@pytest.mark.parametrize('change',['missing','renamed','unrelated-pod','autopay'])
def test_post_teardown_inventory_failure_is_recorded_without_deleting_storage(tmp_path,guard,change):
    f=VolumeFake(tmp_path) if guard=='controller' else Fake(volume_intent()['watchdog_intent'])
    original=f.account
    def account():
        obs=original() if guard=='controller' else observation(original(),f.i['retained_volume'])
        if not f.alive and any(c[0]=='terminate' for c in f.calls):
            if change=='missing':obs['volume_ids']=[];obs['network_volumes']=[]
            if change=='renamed':obs['network_volumes'][0]['name']='changed'
            if change=='unrelated-pod':obs['pods']=[{**f.pod(),'id':'unrelated','name':'unrelated'}]
            if change=='autopay':obs['autopay']=True
        return obs
    f.account=account
    if guard=='controller':f.run();directory=f.directory
    else:directory=tmp_path/'watchdog';f.run(directory)
    result=read_json(directory/'result.json')
    assert result['complete'] and result['retained_storage_verification']=='FAIL' and result['account_guard_violations']
    assert all(c[1]=={'input':{'podId':'owned-pod'}} for c in f.calls if c[0]=='terminate')


def test_watchdog_does_not_arm_on_missing_volume_before_creation(tmp_path):
    w=volume_intent()['watchdog_intent'];f=Fake(w);f.alive=False;path=tmp_path/'watchdog';states=[]
    old_sleep=f.sleep
    def sleep(seconds):
        if (path/'heartbeat.json').exists():states.append(read_json(path/'heartbeat.json')['state'])
        old_sleep(seconds)
    f.sleep=sleep;f.run(path)
    assert states and set(states)=={'TERMINATING'}
    assert read_json(path/'result.json')['retained_storage_verification']=='FAIL'


@pytest.mark.parametrize('size',[224,Decimal('224.0')])
def test_account_volume_observation_is_canonical(monkeypatch,size):
    import probe_provider_deadline as provider
    raw={'myself':{'isAutoPayEnabled':False,'pods':[],'networkVolumes':[{'id':'synthetic-volume','name':'synthetic-cache','size':size,'dataCenterId':'EU-RO-1'}],'clientBalance':100,'currentSpendPerHr':Decimal('.023334')}}
    monkeypatch.setattr(provider,'request',lambda operation:(raw,'a'*64,{'server_epoch':NOW,'request_started_epoch':NOW,'request_completed_epoch':NOW}))
    obs=provider.account();assert type(obs['network_volumes'][0]['size']) is int
    assert digest(obs)


@pytest.mark.parametrize('size',[True,Decimal('224.1'),Decimal('NaN'),-1,'224'])
def test_account_rejects_malformed_volume_size(monkeypatch,size):
    import probe_provider_deadline as provider
    raw={'myself':{'isAutoPayEnabled':False,'pods':[],'networkVolumes':[{'id':'synthetic-volume','name':'synthetic-cache','size':size,'dataCenterId':'EU-RO-1'}],'clientBalance':100,'currentSpendPerHr':Decimal('.023334')}}
    monkeypatch.setattr(provider,'request',lambda operation:(raw,'a'*64,{}))
    with pytest.raises(EvidenceError):provider.account()
