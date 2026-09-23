"""Synthetic prospective price policy; no provider allocation or billing claim."""
from decimal import Decimal,ROUND_CEILING
import pytest
from test_retained_network_volume import volume_intent,VolumeFake
from test_rental_controller import intent
import run_rental_controller as controller
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import rental_plan


def selected():
    v=volume_intent();q=v['quote'];q.update(schema='ovl.rental-quote.v4',rate_margin_percent=110)
    p=v['payload'];rate=(Decimal(q['selected_gpu']['hourly_usd'])+Decimal(p['containerDiskInGb'])*Decimal('.10')/672)*Decimal('1.10')
    w=v['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'quote_sha256':digest(q),
        'hourly_upper_usd':format(rate.quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')})
    return v


def test_prospective_price_margin_keeps_original_deadlines_and_all_reserves():
    v=selected();_,actual=controller.validate(v,digest(v));old=volume_intent()['watchdog_intent']['plan']
    assert actual['input']['hourly_upper_usd']=='0.264655'
    for k in ('provider_terminate_epoch','external_terminate_epoch','request_checkpoint_epoch','protected_reserve_micro_usd'):
        assert actual[k]==old[k]
    for k in ('spent_usd','outstanding_usd','reserved_remaining_usd','maximum_seconds','checkpoint_grace_seconds','external_termination_grace_seconds','billing_slack_seconds'):
        assert actual['input'][k]==old['input'][k]
    assert v['watchdog_intent']['retained_volume']==volume_intent()['watchdog_intent']['retained_volume']


@pytest.mark.parametrize('damage',['legacy-relabel','lower-margin','boolean-margin','wrong-price','underpriced-bound','wrong-storage','no-retained-volume','stale-quote'])
def test_rehashed_invalid_price_selection_fails_before_any_provider_call(tmp_path,damage):
    fake=VolumeFake(tmp_path);v=selected();q=v['quote'];w=v['watchdog_intent']
    if damage=='legacy-relabel':q['schema']='ovl.rental-quote.v3'
    elif damage=='lower-margin':q['rate_margin_percent']=100
    elif damage=='boolean-margin':q['rate_margin_percent']=True
    elif damage=='wrong-price':q['selected_gpu']['hourly_usd']='0.01'
    elif damage=='wrong-storage':q['network_volume_sha256']='0'*64
    elif damage=='no-retained-volume':v['schema']='ovl.rental-controller-intent.v3'
    elif damage=='stale-quote':q['observed_epoch']-=601
    w['plan']=rental_plan({**w['plan']['input'],'quote_sha256':digest(q),
        **({'hourly_upper_usd':'0.01'} if damage=='underpriced-bound' else {})})
    fake.value=v;fake.i=w;fake.refresh()
    with pytest.raises(EvidenceError):fake.run()
    assert fake.calls==[] and fake.writes==0


def test_new_price_selection_still_creates_once_and_terminates_with_original_guard(tmp_path):
    fake=VolumeFake(tmp_path);fake.value=selected();fake.i=fake.value['watchdog_intent'];fake.refresh()
    fake.run();fake.run()
    assert fake.writes==1 and not fake.alive and read_json(fake.directory/'result.json')['complete']
    assert next(c for c in fake.calls if c[0]=='terminate')[2]==fake.i['plan']['provider_terminate_epoch']


def test_live_rate_above_new_bound_stops_attributed_pod_early(tmp_path):
    fake=VolumeFake(tmp_path);fake.value=selected();fake.i=fake.value['watchdog_intent'];fake.refresh()
    fake.volume_change=lambda obs:obs['pods'][0].update(costPerHr='0.265',adjustedCostPerHr='0.265')
    fake.run()
    assert fake.writes==1 and not fake.alive
    assert next(c for c in fake.calls if c[0]=='terminate')[2]<fake.i['plan']['provider_terminate_epoch']
