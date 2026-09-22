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
def test_unrepresentative_or_incomplete_pilot_evidence_rejected(changes):
    e=representative_example();e["phases"]["wikipedia"].update(changes)
    with pytest.raises(EvidenceError):forecast(e)
