"""Synthetic provider values only; no allocation or model acceptance."""
from decimal import Decimal
import json
from email.utils import formatdate
import pytest
import probe_provider_deadline as provider
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import Journal
from test_retained_network_volume import VolumeFake,volume_intent
from test_external_watchdog import NOW,Fake


def raw_account(pod):
    return {'myself':{'isAutoPayEnabled':False,'pods':[pod],
        'networkVolumes':[{'id':'synthetic-volume','name':'synthetic-cache','size':Decimal('224.0'),'dataCenterId':'EU-RO-1'}],
        'clientBalance':100,'currentSpendPerHr':Decimal('.263334')}}


def provider_pod(fake,size):
    pod=fake.pod();pod['volumeInGb']=size
    pod.update(networkVolume={'id':'synthetic-volume','dataCenterId':'EU-RO-1'},volumeMountPath='/workspace',
               desiredStatus='RUNNING',costPerHr=Decimal('.24'),adjustedCostPerHr=Decimal('.24'))
    return pod


@pytest.mark.parametrize('size',[224,Decimal('224.0'),Decimal('2.24E2')])
def test_integral_provider_float_can_be_journaled(tmp_path,monkeypatch,size):
    f=VolumeFake(tmp_path);raw=raw_account(provider_pod(f,size))
    monkeypatch.setattr(provider,'request',lambda operation:(raw,'a'*64,{}))
    obs=provider.account()
    assert type(obs['pods'][0]['volumeInGb']) is int
    assert digest(obs)
    assert not provider.provision_errors(f.i,obs['pods'][0])


@pytest.mark.parametrize('size',[True,'224',None,Decimal('224.1'),Decimal('NaN'),Decimal('Infinity'),-1,2**54])
def test_malformed_provider_volume_remains_strict(tmp_path,monkeypatch,size):
    f=VolumeFake(tmp_path);raw=raw_account(provider_pod(f,size))
    monkeypatch.setattr(provider,'request',lambda operation:(raw,'a'*64,{}))
    with pytest.raises(EvidenceError):provider.account()


def test_shape_rejection_retains_observation_and_specific_fields(tmp_path):
    f=VolumeFake(tmp_path)
    f.volume_change=lambda obs:obs['pods'][0].update(volumeInGb=0)
    f.run();events=Journal(f.directory)._read()
    rejected=[e for e in events if e['kind']=='failure' and e['body'].get('stage')=='resource-shape']
    assert rejected and rejected[0]['body']['fields']==['volumeInGb']
    raw=[e for e in events if e['kind']=='provider-observation' and e['body'].get('validation')=='PENDING']
    assert raw and raw[0]['sequence']<rejected[0]['sequence']
    assert raw[0]['body']['account']['pods'][0]['volumeInGb']==0
    assert not f.alive and f.writes==1
    assert next(c[2] for c in f.calls if c[0]=='terminate')<f.i['plan']['provider_terminate_epoch']
    assert read_json(f.directory/'result.json')['training_admission']=='NOT_RUN'


def test_diagnostic_never_reflects_unknown_exception_text():
    assert provider.diagnostic(EvidenceError('SYNTHETIC_PRIVATE_MARKER'))=={'error_type':'EvidenceError'}


@pytest.mark.parametrize('size',[0,Decimal('-0.0'),2**50])
def test_exact_float_boundaries_remain_canonical(tmp_path,monkeypatch,size):
    f=VolumeFake(tmp_path);raw=raw_account(provider_pod(f,size))
    monkeypatch.setattr(provider,'request',lambda operation:(raw,'a'*64,{}))
    assert provider.account()['pods'][0]['volumeInGb']==int(size)


@pytest.mark.parametrize('size',[True,Decimal('4.0'),None,4.0])
def test_schema_int_field_is_not_float_normalized(tmp_path,monkeypatch,size):
    f=VolumeFake(tmp_path);pod=provider_pod(f,224);pod['containerDiskInGb']=size
    monkeypatch.setattr(provider,'request',lambda operation:(raw_account(pod),'a'*64,{}))
    with pytest.raises(EvidenceError) as error:provider.account()
    assert provider.diagnostic(error.value)['observation_failure']=='container-disk-size'


@pytest.mark.parametrize('guard',['controller','watchdog'])
def test_storage_violation_survives_observation_serialization_failure(tmp_path,guard):
    f=VolumeFake(tmp_path);original=f.account
    failed=False
    def account():
        nonlocal failed
        obs=original()
        if f.alive and not failed:
            failed=True;obs['volume_ids']=[];obs['network_volumes']=[]
            obs['pods'][0]['containerDiskInGb']=Decimal('4.0')
        return obs
    f.account=account
    if guard=='controller':f.run();path=f.directory
    else:
        f.alive=True;path=tmp_path/'watchdog'
        import run_external_watchdog as watchdog
        watchdog.run(path,f.i,digest(f.i),get_account=f.account,provider_request=f.request,
            wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
            clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
    result=read_json(path/'result.json')
    assert result['retained_storage_verification']=='FAIL'
    assert 'retained-volume-identity' in result['account_guard_violations']
    assert not f.alive


@pytest.mark.parametrize('guard',['controller','watchdog'])
@pytest.mark.parametrize('size',[224.0,225.0])
def test_parsed_float_through_bounded_transport_and_guard(tmp_path,monkeypatch,guard,size):
    # Exercise the actual JSON parser and fork/pipe reader, not a normalized
    # account double. All transport and identity values are synthetic.
    f=VolumeFake(tmp_path);original=f.account
    monkeypatch.setattr(provider,'credential',lambda:'synthetic-credential')
    monkeypatch.setattr(provider.time,'time',lambda:f.now)
    class Response:
        status=200;url=provider.ENDPOINT
        @property
        def headers(self):return {'Date':formatdate(f.now,usegmt=True)}
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self,limit):
            v=raw_account(provider_pod(f,size))
            if not f.alive:v['myself']['pods']=[];v['myself']['currentSpendPerHr']=Decimal('.023334')
            return json.dumps({'data':v},default=float).encode()
    class Opener:
        def open(self,request,timeout):
            assert json.loads(request.data)['query']==provider.ACCOUNT
            return Response()
    monkeypatch.setattr(provider,'build_opener',lambda *args:Opener())
    def account():
        f.refresh()
        return provider.account()
    f.account=account
    if guard=='controller':f.run();path=f.directory
    else:
        f.alive=True;path=tmp_path/'watchdog'
        import run_external_watchdog as watchdog
        watchdog.run(path,f.i,digest(f.i),get_account=f.account,provider_request=f.request,
            wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
            clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
    events=Journal(path)._read();first=next(c[2] for c in f.calls if c[0]=='terminate')
    if size==224.0:
        deadline=f.i['plan']['provider_terminate_epoch' if guard=='controller' else 'external_terminate_epoch']
        assert first==deadline
        assert not [e for e in events if e['kind']=='failure']
    else:
        assert first==NOW
        assert any(e['body'].get('fields')==['volumeInGb'] for e in events)
    assert not f.alive and read_json(path/'result.json')['complete']
    if guard=='controller':assert f.writes==1


@pytest.mark.parametrize('guard',['controller','watchdog'])
def test_unverified_account_survives_guard_restart(tmp_path,guard):
    f=VolumeFake(tmp_path);original=f.account;failed=False;interrupted=False
    def account():
        nonlocal failed
        if f.alive and not failed:
            failed=True;raise EvidenceError('invalid provider pod volume size')
        return original()
    f.account=account
    original_request=f.provider if guard=='controller' else f.request
    def request(operation,variables=None):
        nonlocal interrupted
        result=original_request(operation,variables)
        if operation=='terminate' and not interrupted:
            interrupted=True;raise KeyboardInterrupt('synthetic guard process interruption')
        return result
    if guard=='controller':f.provider=request;run=f.run;path=f.directory
    else:
        f.alive=True;path=tmp_path/'watchdog'
        import run_external_watchdog as watchdog
        def run():
            watchdog.run(path,f.i,digest(f.i),get_account=f.account,provider_request=request,
                wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
                clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
    with pytest.raises(KeyboardInterrupt):run()
    assert not f.alive and not (path/'result.json').exists()
    run();result=read_json(path/'result.json')
    assert result['retained_storage_verification']=='FAIL'
    assert 'provider-observation-unverified' in result['account_guard_violations']
    if guard=='controller':assert f.writes==1
