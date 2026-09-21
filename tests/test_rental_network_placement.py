"""Network placement is a pinned request, never measured throughput evidence."""
from copy import deepcopy
import pytest
from test_rental_cloud_quote import selected
from test_rental_controller import RentalFake
import run_rental_controller as controller
from ovl_pipeline.canonical import EvidenceError,digest,read_json


def network(cloud='COMMUNITY'):
    value=selected(cloud);value['schema']='ovl.rental-controller-intent.v4'
    value['payload'].update(minDownload=500,minUpload=100)
    return value


@pytest.mark.parametrize('cloud',['COMMUNITY','SECURE'])
def test_network_request_preserves_lifetime_and_reconciles_once(tmp_path,cloud):
    old=selected(cloud);value=network(cloud)
    assert value['watchdog_intent']==old['watchdog_intent'] and value['quote']==old['quote']
    fake=RentalFake(tmp_path);fake.value=value;fake.i=value['watchdog_intent'];fake.refresh()
    fake.run();fake.run()
    assert fake.writes==1 and not fake.alive
    sent=next(c[1]['input'] for c in fake.calls if c[0]=='create')
    assert sent==value['payload']
    assert next(c for c in fake.calls if c[0]=='terminate')[2]==fake.i['plan']['provider_terminate_epoch']
    result=read_json(fake.directory/'result.json')
    assert {'minDownload','minUpload'}<=set(result['provider_requested_only_fields'])
    assert result['runtime_identity_admission']==result['training_admission']=='NOT_RUN'
    assert result['automatic_provider_termination']=='UNVERIFIED'


@pytest.mark.parametrize('key',['minDownload','minUpload'])
@pytest.mark.parametrize('bad',[None,False,True,0,-1,100001,'500',1.5,[],{}])
def test_bad_network_value_never_calls_provider(tmp_path,key,bad):
    fake=RentalFake(tmp_path);fake.value=network();fake.value['payload'][key]=bad
    with pytest.raises((EvidenceError,ValueError)):fake.run()
    assert fake.calls==[] and fake.writes==0


@pytest.mark.parametrize('key',['minDownload','minUpload'])
def test_missing_minimum_fails_before_any_provider_call(tmp_path,key):
    fake=RentalFake(tmp_path);fake.value=network();del fake.value['payload'][key]
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[]


@pytest.mark.parametrize('version',['v1','v2','v3'])
def test_historical_schema_does_not_accept_network_fields(version):
    value=network();value['schema']='ovl.rental-controller-intent.'+version
    with pytest.raises(EvidenceError):controller.validate(value,digest(value))


def test_changed_selection_or_identity_cannot_borrow_original_pin():
    value=network();pin=digest(value)
    for key,replacement in [('minDownload',501),('minUpload',101),('terminateAfter','2099-01-01T00:00:00Z'),('name','another-rental')]:
        changed=deepcopy(value);changed['payload'][key]=replacement
        with pytest.raises(EvidenceError,match='selected pin'):controller.validate(changed,pin)


@pytest.mark.parametrize('key,replacement',[
    ('terminateAfter','2099-01-01T00:00:00Z'),('name','another-rental'),('env',{}),
    ('allowedCudaVersions',['12.8']),('cloudType','ALL'),('gpuCount',2),
])
def test_network_schema_preserves_existing_strict_creation_gates(tmp_path,key,replacement):
    fake=RentalFake(tmp_path);fake.value=network();fake.value['payload'][key]=replacement
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[]
