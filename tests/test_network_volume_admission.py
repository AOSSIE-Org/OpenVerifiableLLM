"""Synthetic network attachment integration; no provider calls or paid resources."""
import hashlib
import json
from decimal import Decimal, ROUND_CEILING
import pytest
import retained_volume
import run_external_watchdog as watchdog
from ovl_pipeline.canonical import digest,read_json
from ovl_pipeline.supervision import Journal,rental_plan
from test_retained_network_volume import VolumeFake


def test_selected_twenty_gb_container_and_zero_local_volume(tmp_path):
    f=VolumeFake(tmp_path);w=f.i;q=f.value['quote']
    for payload in (w['payload'],f.value['payload']):payload['containerDiskInGb']=20
    f.value['payload'].update(gpuTypeId='NVIDIA GeForce RTX 5090',minVcpuCount=4,minMemoryInGb=16)
    q['selected_gpu'].update(id='NVIDIA GeForce RTX 5090',hourly_usd='0.99')
    q['catalog_response']=json.dumps({'gpus':[{'id':'NVIDIA GeForce RTX 5090','secure':True,'maxCount':{'secure':1},'price':{'secure':'0.99'}}]})
    q['catalog_response_sha256']=hashlib.sha256(q['catalog_response'].encode()).hexdigest()
    rate=((Decimal('.99')+Decimal(20)*Decimal('.10')/672)*Decimal('1.25')).quantize(Decimal('.000001'),rounding=ROUND_CEILING)
    w['plan']=rental_plan({**w['plan']['input'],'quote_sha256':digest(q),'hourly_upper_usd':str(rate)})
    f.refresh();f.run()
    assert f.writes==1 and not f.alive
    assert w['payload']['volumeInGb']==0 and w['payload']['gpuCount']==1
    assert w['retained_volume']['size_gb']==224 and w['retained_volume']['reserved_usd']=='4.480128'
    assert rate==Decimal('1.241221')
    assert next(x[2] for x in f.calls if x[0]=='terminate')==w['plan']['provider_terminate_epoch']
    assert not [e for e in Journal(f.directory)._read() if e['kind']=='failure']


@pytest.mark.parametrize('guard',['controller','watchdog'])
def test_lagged_rate_failure_stays_latched_after_restart(tmp_path,guard):
    f=VolumeFake(tmp_path);path=f.directory if guard=='controller' else tmp_path/'watchdog'
    original_account=f.account;original_sleep=f.sleep;settled=False
    def account():
        obs=original_account()
        if not f.alive and any(c[0]=='terminate' for c in f.calls) and not settled:
            obs['account_hourly_usd']='0.263334'
        return obs
    def sleep(seconds):
        if path.exists() and not settled:
            events=Journal(path)._read()
            if any(e['kind']=='failure' and e['body'].get('reasons')==['account-rate'] for e in events):
                raise KeyboardInterrupt('synthetic stop after durable rate failure')
        original_sleep(seconds)
    f.account=account;f.sleep=sleep
    if guard=='controller':run=f.run
    else:
        f.alive=True
        def run():
            watchdog.run(path,f.i,digest(f.i),get_account=f.account,provider_request=f.request,
                wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
                clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
    with pytest.raises(KeyboardInterrupt):run()
    assert not f.alive and not (path/'result.json').exists()
    assert not retained_volume.baseline_valid(f.i,account())
    settled=True;run();result=read_json(path/'result.json')
    assert result['complete'] and result['retained_storage_verification']=='FAIL'
    assert 'account-rate' in result['account_guard_violations']
    assert retained_volume.baseline_valid(f.i,account())
    if guard=='controller':assert f.writes==1
