"""Future cloud selection, with unchanged cost/lifetime guards and no live writes."""
from copy import deepcopy
from decimal import Decimal,ROUND_CEILING
import hashlib
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_rental_controller as controller
from test_rental_controller import intent,RentalFake
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import rental_plan


def selected(cloud='COMMUNITY'):
    value=intent();value['schema']='ovl.rental-controller-intent.v3'
    value['payload'].update(cloudType=cloud,allowedCudaVersions=['13.0'])
    q=value['quote'];q['schema']='ovl.rental-quote.v2'
    q['selected_gpu']={'id':value['payload']['gpuTypeId'],'cloud':cloud,
                       'hourly_usd':'0.69' if cloud=='COMMUNITY' else '0.99'}
    catalog={'gpus':[{'id':value['payload']['gpuTypeId'],'secure':True,'community':True,
                     'maxCount':{'secure':1,'community':1},'price':{'secure':'0.99','community':'0.69'}}]}
    set_catalog(q,catalog)
    p=value['payload'];hourly=((Decimal(q['selected_gpu']['hourly_usd'])+
        (Decimal(p['containerDiskInGb'])*Decimal('.10')+Decimal(p['volumeInGb'])*Decimal('.20'))/672)*Decimal('1.25'))
    v={**value['watchdog_intent']['plan']['input'],'quote_sha256':digest(q),
       'hourly_upper_usd':format(hourly.quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')}
    value['watchdog_intent']['plan']=rental_plan(v)
    return value


def set_catalog(q,catalog):
    q['catalog_response']=json.dumps(catalog)
    q['catalog_response_sha256']=hashlib.sha256(q['catalog_response'].encode()).hexdigest()


def repin(value):
    w=value['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'quote_sha256':digest(value['quote'])})


@pytest.mark.parametrize('cloud',['SECURE','COMMUNITY'])
def test_selected_lane_uses_its_price_with_original_rate_margin_and_deadlines(cloud):
    old=intent();value=selected(cloud);w,p=controller.validate(value,digest(value))
    for k in ('provider_terminate_epoch','external_terminate_epoch','request_checkpoint_epoch','protected_reserve_micro_usd'):
        assert p[k]==old['watchdog_intent']['plan'][k]
    assert value['quote']['rate_margin_percent']==125
    assert p['input']['hourly_upper_usd']==('0.863245' if cloud=='COMMUNITY' else '1.238245')


@pytest.mark.parametrize('cloud',['SECURE','COMMUNITY'])
def test_new_selection_runs_one_fake_creation_and_original_verified_teardown(tmp_path,cloud):
    fake=RentalFake(tmp_path);fake.value=selected(cloud);fake.i=fake.value['watchdog_intent'];fake.refresh()
    fake.run();fake.run()
    assert fake.writes==1 and not fake.alive
    assert next(c for c in fake.calls if c[0]=='create')[1]['input']['cloudType']==cloud
    assert next(c for c in fake.calls if c[0]=='terminate')[2]==fake.i['plan']['provider_terminate_epoch']
    assert read_json(fake.directory/'result.json')['complete']


@pytest.mark.parametrize('damage',[
    'cloud-crossed','wrong-price','unsupported-cloud','catalog-unavailable','catalog-count-zero',
    'catalog-count-bool','duplicate-gpu','wrong-gpu','stale-quote','reduced-rate-margin',
    'underpriced-bound','missing-cuda','unknown-cuda','historical-quote','changed-catalog-bytes',
])
def test_mismatched_or_cheaper_unselected_quote_fails_before_any_provider_call(tmp_path,damage):
    fake=RentalFake(tmp_path);value=selected();q=value['quote'];catalog=json.loads(q['catalog_response'])
    if damage=='cloud-crossed':value['payload']['cloudType']='SECURE'
    elif damage=='wrong-price':q['selected_gpu']['hourly_usd']='0.01'
    elif damage=='unsupported-cloud':q['selected_gpu']['cloud']='SPOT'
    elif damage=='catalog-unavailable':catalog['gpus'][0]['community']=False
    elif damage=='catalog-count-zero':catalog['gpus'][0]['maxCount']['community']=0
    elif damage=='catalog-count-bool':catalog['gpus'][0]['maxCount']['community']=True
    elif damage=='duplicate-gpu':catalog['gpus'].append(deepcopy(catalog['gpus'][0]))
    elif damage=='wrong-gpu':value['payload']['gpuTypeId']='another GPU'
    elif damage=='stale-quote':q['observed_epoch']-=601
    elif damage=='reduced-rate-margin':q['rate_margin_percent']=100
    elif damage=='missing-cuda':del value['payload']['allowedCudaVersions']
    elif damage=='unknown-cuda':value['payload']['allowedCudaVersions']=['12.8']
    elif damage=='historical-quote':value['quote']=intent()['quote']
    if damage.startswith('catalog-') or damage=='duplicate-gpu':set_catalog(q,catalog)
    if damage=='changed-catalog-bytes':q['catalog_response']+=' '
    repin(value)
    if damage=='underpriced-bound':
        w=value['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'hourly_upper_usd':'0.01'})
    fake.value=value;fake.i=value['watchdog_intent'];fake.refresh()
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[] and fake.writes==0


@pytest.mark.parametrize('version',['v1','v2'])
def test_historical_intents_cannot_be_relabelled_community_or_new_quote(version):
    value=intent();value['schema']='ovl.rental-controller-intent.'+version
    if version=='v2':value['payload']['allowedCudaVersions']=['13.0']
    controller.validate(value,digest(value))
    value['payload']['cloudType']='COMMUNITY'
    with pytest.raises(EvidenceError):controller.validate(value,digest(value))
    value['payload']['cloudType']='SECURE';value['quote']=selected('SECURE')['quote'];repin(value)
    with pytest.raises(EvidenceError):controller.validate(value,digest(value))
