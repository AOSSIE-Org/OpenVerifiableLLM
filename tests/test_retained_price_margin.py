"""Synthetic prospective price policy; no provider allocation or billing claim."""
from decimal import Decimal,ROUND_CEILING
import pytest
from test_retained_network_volume import volume_intent,VolumeFake
from test_rental_controller import intent
import run_rental_controller as controller
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import rental_plan


def selected(version='v4'):
    v=volume_intent();q=v['quote'];q.update(schema='ovl.rental-quote.'+version,rate_margin_percent=110 if version=='v4' else 102)
    p=v['payload'];rate=(Decimal(q['selected_gpu']['hourly_usd'])+Decimal(p['containerDiskInGb'])*Decimal('.10')/672)*Decimal(q['rate_margin_percent'])/100
    w=v['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'quote_sha256':digest(q),
        'hourly_upper_usd':format(rate.quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')})
    return v


@pytest.mark.parametrize('version,expected',[('v4','0.264655'),('v5','0.245408')])
def test_prospective_price_margin_keeps_original_deadlines_and_all_reserves(version,expected):
    v=selected(version);_,actual=controller.validate(v,digest(v));old=volume_intent()['watchdog_intent']['plan']
    assert actual['input']['hourly_upper_usd']==expected
    for k in ('provider_terminate_epoch','external_terminate_epoch','request_checkpoint_epoch','protected_reserve_micro_usd'):
        assert actual[k]==old[k]
    for k in ('spent_usd','outstanding_usd','reserved_remaining_usd','maximum_seconds','checkpoint_grace_seconds','external_termination_grace_seconds','billing_slack_seconds'):
        assert actual['input'][k]==old['input'][k]
    assert v['watchdog_intent']['retained_volume']==volume_intent()['watchdog_intent']['retained_volume']


@pytest.mark.parametrize('damage',['legacy-relabel','lower-margin','boolean-margin','wrong-price','underpriced-bound','wrong-storage','no-retained-volume','stale-quote'])
@pytest.mark.parametrize('version',['v4','v5'])
def test_rehashed_invalid_price_selection_fails_before_any_provider_call(tmp_path,damage,version):
    fake=VolumeFake(tmp_path);v=selected(version);q=v['quote'];w=v['watchdog_intent']
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


@pytest.mark.parametrize('version',['v4','v5'])
def test_new_price_selection_still_creates_once_and_terminates_with_original_guard(tmp_path,version):
    fake=VolumeFake(tmp_path);fake.value=selected(version);fake.i=fake.value['watchdog_intent'];fake.refresh()
    fake.run();fake.run()
    assert fake.writes==1 and not fake.alive and read_json(fake.directory/'result.json')['complete']
    assert next(c for c in fake.calls if c[0]=='terminate')[2]==fake.i['plan']['provider_terminate_epoch']


@pytest.mark.parametrize('version',['v4','v5'])
@pytest.mark.parametrize('rate_field',['costPerHr','adjustedCostPerHr','account_hourly_usd'])
def test_live_rate_above_new_bound_stops_attributed_pod_early(tmp_path,version,rate_field):
    fake=VolumeFake(tmp_path);fake.value=selected(version);fake.i=fake.value['watchdog_intent'];fake.refresh()
    def change(obs):
        if rate_field=='account_hourly_usd':obs[rate_field]='0.5'
        else:obs['pods'][0][rate_field]=str(Decimal(fake.i['plan']['input']['hourly_upper_usd'])+Decimal('.000001'))
    fake.volume_change=change
    fake.run()
    assert fake.writes==1 and not fake.alive
    assert next(c for c in fake.calls if c[0]=='terminate')[2]<fake.i['plan']['provider_terminate_epoch']


@pytest.mark.parametrize('version,margin',[('v4',102),('v5',110)])
def test_prospective_version_cannot_borrow_other_margin(version,margin):
    v=selected(version);v['quote']['rate_margin_percent']=margin
    v['watchdog_intent']['plan']=rental_plan({**v['watchdog_intent']['plan']['input'],'quote_sha256':digest(v['quote'])})
    with pytest.raises(EvidenceError,match='rate margin'):controller.validate(v,digest(v))


@pytest.mark.parametrize('field',['costPerHr','adjustedCostPerHr','account_hourly_usd'])
@pytest.mark.parametrize('increment',['0','.000001','NaN','Infinity','missing'])
def test_v5_watchdog_price_boundary_and_invalid_observations(tmp_path,field,increment):
    from test_external_watchdog import Fake
    from test_retained_network_volume import observation
    import retained_volume
    w=selected('v5')['watchdog_intent'];f=Fake(w);original=f.account
    bound=Decimal(w['plan']['input']['hourly_upper_usd'])
    def account():
        obs=observation(original(),w['retained_volume'])
        if obs['pods']:
            obj=obs if field=='account_hourly_usd' else obs['pods'][0]
            if increment=='missing':del obj[field]
            else:
                value=bound+Decimal(increment)
                if field=='account_hourly_usd':value+=retained_volume.hourly(w['retained_volume'])
                obj[field]=str(value)
        return obs
    f.account=account;f.run(tmp_path/'guard')
    first=next(c for c in f.calls if c[0]=='terminate')
    if increment=='0':assert first[2]==w['plan']['external_terminate_epoch']
    else:assert first[2]<w['plan']['external_terminate_epoch']
    assert not f.alive and read_json(tmp_path/'guard/result.json')['complete']


def test_repricing_cannot_adopt_an_existing_rental_journal(tmp_path):
    f=VolumeFake(tmp_path);f.value=selected('v4');f.i=f.value['watchdog_intent'];f.refresh();f.run()
    before=list(f.calls);f.value=selected('v5');f.i=f.value['watchdog_intent'];f.refresh()
    with pytest.raises(EvidenceError):f.run()
    assert f.calls==before and f.writes==1


def test_affordability_limited_new_quote_is_distinct_prospective_plan():
    values=[]
    for version in ('v4','v5'):
        v=selected(version);old=v['watchdog_intent']['plan']['input']
        plan=rental_plan({**old,'allowance_usd':'1','maximum_seconds':20000})
        values.append(plan)
        assert plan['input']['maximum_seconds']==20000
        assert plan['maximum_charge_micro_usd']<=1000000
    assert values[1]['provider_terminate_epoch']>values[0]['provider_terminate_epoch']
    assert values[1]['input']['quote_sha256']!=values[0]['input']['quote_sha256']
