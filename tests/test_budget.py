import copy
import pytest

from ovl_pipeline.budget import forecast, money
from ovl_pipeline.canonical import EvidenceError


def example():
    phase = {"targets": 1_000_000, "training_completed": 0, "replay_completed": 0,
             "measured_targets": 1_000_000, "measured_ms": 600_000,
             "warmup_excluded": True, "overhead_included": True,
             "measurement_sha256": "a" * 64, "recipe_sha256": "b" * 64, "stream_sha256": "c" * 64}
    return {"schema": "ovl.cost-forecast-input.v1", "spent_usd": "10", "committed_future_usd": "5",
            "fixed_remaining_usd": "5", "hourly_usd": "1.2",
            "phases": {"wikipedia": copy.deepcopy(phase), "conversation": copy.deepcopy(phase)}}


def test_full_replay_and_margin_are_charged():
    r = forecast(example())
    # Two phases * (10min training + 10min replay) * 1.25 = 50min.
    assert r["remaining_compute_micro_usd"] == 1_000_000
    assert r["projected_total_micro_usd"] == 21_000_000
    assert r["result"] == "FITS_OPERATING_LIMIT"
    assert r["protected_reserve_micro_usd"] == 10_000_000
    assert r["cap_micro_usd"] == 130_000_000
    assert r["operating_limit_micro_usd"] == 120_000_000
    assert r["provider_guard"] == "NOT_RUN"
    e = example();e["phases"]["wikipedia"]["training_completed"] = 1_000_000
    assert forecast(e)["remaining_compute_micro_usd"] == 750_000
    e["phases"]["wikipedia"]["replay_completed"] = 1_000_000
    assert forecast(e)["remaining_compute_micro_usd"] == 500_000


def test_stop_at_operating_limit_preserves_export_funds():
    e = example();e["spent_usd"] = "109"
    assert forecast(e)["projected_total_micro_usd"] == 120_000_000
    assert forecast(e)["result"] == "FITS_OPERATING_LIMIT"
    e["spent_usd"] = "109.000001"
    assert forecast(e)["result"] == "STOP"
    e["spent_usd"] = "130"
    assert forecast(e)["maximum_affordable_compute_ms"] == 0


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "0.0000001", "01", 1.0, True, "1e2"])
def test_money_never_uses_float_or_rounds_down(value):
    with pytest.raises(EvidenceError): money(value)


@pytest.mark.parametrize("key,value", [("targets", 0), ("measured_targets", 0), ("measured_ms", 599999),
    ("training_completed", 1000001), ("replay_completed", 1), ("warmup_excluded", False),
    ("overhead_included", False), ("measurement_sha256", "absent")])
def test_missing_or_short_measurements_fail(key, value):
    e=example();e["phases"]["wikipedia"][key]=value
    with pytest.raises(EvidenceError): forecast(e)


def test_unknown_fields_and_sampled_phase_fail():
    e=example();del e["phases"]["conversation"]
    with pytest.raises(EvidenceError):forecast(e)
    e=example();e["phases"]["wikipedia"]["audit_fraction"] = "0.01"
    with pytest.raises(EvidenceError):forecast(e)


def representative_example():
    e=example();e["schema"]="ovl.cost-forecast-input.v3"
    for p in e["phases"].values():
        p.pop("targets");p.pop("measured_targets")
        p.update(updates=1000,measured_full_batch_updates=100,measured_updates=100,
                 schedule_sha256="d"*64,replay_sha256="e"*64,
                 eligible_duration_for_forecast=True,replay_measured_ms=1200000,
                 measured_checkpoints=10,measured_checkpoint_every=10,
                 production_checkpoint_every=10,production_checkpoints=100)
    return e


def test_slower_complete_replay_prices_both_paths():
    r=forecast(representative_example())
    # Two phases * 2000 updates * 12 sec/update * 1.25 = 60000 sec.
    assert r["remaining_compute_micro_usd"]==20_000_000
    assert r["schema"]=="ovl.cost-forecast.v3"
    e=representative_example()
    for p in e["phases"].values():p["replay_measured_ms"]=300000
    assert forecast(e)["remaining_compute_micro_usd"]==10_000_000


def test_representative_forecast_counts_prior_liabilities_at_new_limit():
    e=representative_example();e['spent_usd']='90'
    assert forecast(e)['projected_total_micro_usd']==120_000_000
    assert forecast(e)['result']=='FITS_OPERATING_LIMIT'
    e['committed_future_usd']='5.000001'
    assert forecast(e)['result']=='STOP'
    e['operating_limit_micro_usd']=130_000_000
    with pytest.raises(EvidenceError):forecast(e)


@pytest.mark.parametrize("changes",[
    {"eligible_duration_for_forecast":False},
    {"replay_sha256":""}, {"replay_measured_ms":0},
    {"measured_updates":99}, {"measured_checkpoints":9},
    {"production_checkpoints":99},
    {"production_checkpoints":101},  # Extra recovery checkpoint must be priced too.
    {"measured_checkpoint_every":20,"measured_checkpoints":5},
])
@pytest.mark.parametrize('version',['v3','v4'])
def test_unrepresentative_or_incomplete_pilot_evidence_rejected(changes,version):
    e=representative_example();e['schema']='ovl.cost-forecast-input.'+version;e["phases"]["wikipedia"].update(changes)
    with pytest.raises(EvidenceError):forecast(e)


def test_directional_forecast_prices_full_record_and_replay_with_margin():
    e=representative_example();legacy=forecast(e);e['schema']='ovl.cost-forecast-input.v4'
    r=forecast(e)
    # Two phases * (6000s recording + 12000s replay) * 1.25.
    assert r['remaining_compute_micro_usd']==15_000_000
    assert r['schema']=='ovl.cost-forecast.v4'
    assert legacy['remaining_compute_micro_usd']==20_000_000
    e['phases']['wikipedia']['training_completed']=1000
    assert forecast(e)['remaining_compute_micro_usd']==12_500_000
    e['phases']['wikipedia']['replay_completed']=1000
    assert forecast(e)['remaining_compute_micro_usd']==7_500_000


def test_directional_forecast_charges_fixed_work_and_prior_liabilities_at_same_limit():
    e=representative_example();e['schema']='ovl.cost-forecast-input.v4';e['spent_usd']='95'
    assert forecast(e)['projected_total_micro_usd']==120_000_000
    assert forecast(e)['protected_reserve_micro_usd']==10_000_000
    e['fixed_remaining_usd']='5.000001'
    assert forecast(e)['result']=='STOP'
    e['schema']='ovl.cost-forecast-input.v3'
    assert forecast(e)['remaining_compute_micro_usd']==20_000_000


def test_directional_forecast_rounds_each_trajectory_and_margin_up():
    from fractions import Fraction
    from math import ceil
    e=representative_example();e['schema']='ovl.cost-forecast-input.v4'
    for phase in e['phases'].values():
        phase.update(measured_full_batch_updates=97,measured_ms=600001,replay_measured_ms=1200001)
    result=forecast(e)
    one=ceil((ceil(Fraction(1000*600001,97))+ceil(Fraction(1000*1200001,97)))*Fraction(5,4))
    assert all(p['remaining_ms_with_margin']==one for p in result['phases'].values())
    assert result['remaining_compute_micro_usd']==ceil(Fraction(2*one*1200000,3600000))


@pytest.mark.parametrize('version,completed,expected', [
    ('v1',0,'f449b26867adb75ab924c82ee905b13097a6fe2c19355f2dce321718ade49c9f'),
    ('v1',317,'65c73bee5eddbefe6eb92e5839d63be714f497621eb7498f4572baf76d5da2b4'),
    ('v1',1000,'16da07389aeaccdf6d15ca020c2f9651727668658f895ecfee25228c721d9ab1'),
    ('v2',0,'2cfaf84ce031537e6216dff376db3840c40fe90e6a50bfd3886e0a2e03b9c20e'),
    ('v2',317,'c606b9f86d3e23414d9722a5f00066e35762089044499d07c2c6cf6515155f54'),
    ('v2',1000,'7dc96a2d1770336527156e263eedbd01ec42c89f76d2698e74fedca749d61263'),
    ('v3',0,'6e74347b56368d422cdedaef99e6fd92d26906c1f1d89b91998a332f8bc8b143'),
    ('v3',317,'9fc1468247997128a0aa53be2c8250e35bd01453f65fa32104278541238bb7ce'),
    ('v3',1000,'b7462bd65310bcb9c9445a5091d816db30696711d6765363bb4be33170f5f0d0'),
])
def test_historical_forecast_canonical_bytes_remain_compatible(version,completed,expected):
    # Complete output digests produced by the pre-v4 implementation at
    # 363c3db5bd2ddd7e96becb48b7357c2b1e5142c6, including asymmetric work.
    from ovl_pipeline.canonical import digest
    e=example() if version=='v1' else representative_example()
    e['schema']='ovl.cost-forecast-input.'+version
    for p in e['phases'].values():
        if version=='v2':
            for k in ('replay_sha256','eligible_duration_for_forecast','measured_updates','replay_measured_ms',
                      'measured_checkpoints','measured_checkpoint_every','production_checkpoint_every','production_checkpoints'):
                p.pop(k)
        p['training_completed']=completed;p['replay_completed']=completed//3
    assert digest(forecast(e))==expected


def test_faster_measured_replay_remains_fully_priced_with_asymmetric_remaining_work():
    from fractions import Fraction
    from math import ceil
    e=representative_example();e['schema']='ovl.cost-forecast-input.v4'
    for p in e['phases'].values():
        p.update(measured_full_batch_updates=97,measured_ms=600001,replay_measured_ms=150003,
                 training_completed=317,replay_completed=19)
    result=forecast(e)
    one=ceil((ceil(Fraction(683*600001,97))+ceil(Fraction(981*150003,97)))*Fraction(5,4))
    assert all(p['remaining_ms_with_margin']==one and p['remaining_updates_including_replay']==1664
               for p in result['phases'].values())
