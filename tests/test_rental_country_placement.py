"""Country selection changes placement requests without granting acceptance."""
from copy import deepcopy
import pytest
from test_rental_network_placement import network
from test_rental_controller import RentalFake
import run_rental_controller as controller
from ovl_pipeline.canonical import EvidenceError,digest,read_json


def country(cloud='COMMUNITY'):
    value=network(cloud)
    value['schema']='ovl.rental-controller-intent.v5'
    value['payload']['countryCode']='CA'
    return value


@pytest.mark.parametrize('cloud',['COMMUNITY','SECURE'])
def test_pinned_country_preserves_one_shot_creation_and_termination(tmp_path,cloud):
    old=network(cloud);value=country(cloud)
    assert value['watchdog_intent']==old['watchdog_intent']
    assert value['quote']==old['quote']
    fake=RentalFake(tmp_path);fake.value=value;fake.i=value['watchdog_intent'];fake.refresh()
    fake.run();fake.run()
    assert fake.writes==1 and not fake.alive
    assert next(c[1]['input'] for c in fake.calls if c[0]=='create')==value['payload']
    assert next(c for c in fake.calls if c[0]=='terminate')[2]==fake.i['plan']['provider_terminate_epoch']
    result=read_json(fake.directory/'result.json')
    assert {'countryCode','minDownload','minUpload'}<=set(result['provider_requested_only_fields'])
    assert result['automatic_provider_termination']=='UNVERIFIED'
    assert result['runtime_identity_admission']==result['training_admission']=='NOT_RUN'


@pytest.mark.parametrize('bad',[None,False,7,[],{},'', 'ca','CAN','CA,US','C1','CＡ','CA\n',' CA'])
def test_malformed_country_fails_before_provider_calls(tmp_path,bad):
    fake=RentalFake(tmp_path);fake.value=country();fake.value['payload']['countryCode']=bad
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[] and fake.writes==0


def test_missing_country_fails_before_provider_calls(tmp_path):
    fake=RentalFake(tmp_path);fake.value=country();del fake.value['payload']['countryCode']
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[]


@pytest.mark.parametrize('version',['v1','v2','v3','v4'])
def test_old_schema_cannot_borrow_country_selection(version):
    value=country();value['schema']='ovl.rental-controller-intent.'+version
    with pytest.raises(EvidenceError):controller.validate(value,digest(value))


def test_changed_country_cannot_borrow_original_pin():
    value=country();changed=deepcopy(value);changed['payload']['countryCode']='US'
    with pytest.raises(EvidenceError,match='selected pin'):controller.validate(changed,digest(value))


@pytest.mark.parametrize('key,replacement',[
    ('name','another-rental'),('terminateAfter','2099-01-01T00:00:00Z'),
    ('gpuCount',2),('env',{}),('allowedCudaVersions',['12.8']),
    ('cloudType','ALL'),('minDownload',False),('minUpload',0),
])
def test_country_schema_preserves_existing_strict_gates(tmp_path,key,replacement):
    fake=RentalFake(tmp_path);fake.value=country();fake.value['payload'][key]=replacement
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[]
