"""Directional time admission with unchanged conservative money and trust gates."""
from fractions import Fraction
from math import ceil

import pytest

from ovl_pipeline.budget import forecast
from ovl_pipeline.canonical import EvidenceError, digest
from ovl_pipeline.production_chain import schedule
from test_production_run_coordinator import configured, registration_fixture


def required_seconds(r, field):
    # Independent rational arithmetic: round measured extrapolation up to a
    # millisecond, then reserve 25 percent and round each phase up to a second.
    return sum(ceil(Fraction(ceil(Fraction(p['updates'] * p[field],
                                        p['measured_full_batch_updates'])) * 5, 4000))
               for p in r['forecast_input']['phases'].values())


@pytest.mark.parametrize('record_ms,replay_ms', [(600001, 150003), (600001, 1200003), (600001, 600001)])
def test_each_direction_uses_its_measured_rate_without_reducing_money(tmp_path, monkeypatch, record_ms, replay_ms):
    factory, *_ = configured(tmp_path, monkeypatch)
    _, r, _, provider, parents = registration_fixture(tmp_path, monkeypatch)
    for p in r['forecast_input']['phases'].values():
        p['measured_ms'], p['replay_measured_ms'] = record_ms, replay_ms
    expected = forecast(r['forecast_input'])
    with factory() as run:
        run.qualified = {k: parents[k] for k in ('pilot_records', 'pilot_replays')}
        timing = run.selection['timing']
        publication = len(schedule(r)) * timing['publication_policy']['boundary_seconds']
        timing['record_seconds'] = required_seconds(r, 'measured_ms') + publication + timing['record_fixed_seconds']
        timing['replay_seconds'] = required_seconds(r, 'replay_measured_ms') + timing['replay_fixed_seconds']
        assert run.forecast_window(r) == expected
        # This helper neither signs nor admits production; public registration
        # and the original whole-rental remaining-time gate still follow it.
        assert provider.commits == 0 and run.authenticated is None
        for direction in ('record', 'replay'):
            timing[direction + '_seconds'] -= 1
            with pytest.raises(EvidenceError, match='phase budgets'):
                run.forecast_window(r)
            timing[direction + '_seconds'] += 1
        timing['publication_policy']['boundary_seconds'] += 1
        with pytest.raises(EvidenceError, match='phase budgets'):
            run.forecast_window(r)


def test_partial_batches_cannot_shorten_full_production_deadlines(tmp_path, monkeypatch):
    factory, *_ = configured(tmp_path, monkeypatch)
    _, r, _, _, parents = registration_fixture(tmp_path, monkeypatch)
    with factory() as run:
        run.qualified = {k: parents[k] for k in ('pilot_records', 'pilot_replays')}
        timing = run.selection['timing']
        timing['record_seconds'] = required_seconds(r, 'measured_ms') + len(schedule(r)) + timing['record_fixed_seconds']
        timing['replay_seconds'] = required_seconds(r, 'replay_measured_ms') + timing['replay_fixed_seconds']
        assert run.forecast_window(r) == forecast(r['forecast_input'])
        for p in r['forecast_input']['phases'].values():
            assert p['measured_full_batch_updates'] > 1
            p['measured_full_batch_updates'] //= 2
        with pytest.raises(EvidenceError, match='phase budgets'):
            run.forecast_window(r)


@pytest.mark.parametrize('direction', ['record', 'replay'])
def test_fixed_reserve_cannot_omit_observed_setup_margin(tmp_path, monkeypatch, direction):
    factory, *_ = configured(tmp_path, monkeypatch)
    _, r, _, _, parents = registration_fixture(tmp_path, monkeypatch)
    with factory() as run:
        run.qualified = {k: parents[k] for k in ('pilot_records', 'pilot_replays')}
        # Both corpus setup measurements are 100 seconds; 250 seconds is the
        # combined minimum after margin. More fixed work may need more reserve.
        run.selection['timing'][direction + '_fixed_seconds'] = 249
        with pytest.raises(EvidenceError, match='fixed phase reserve'):
            run.forecast_window(r)


@pytest.mark.parametrize('field', ['measured_ms', 'replay_measured_ms'])
def test_registration_rejects_a_forged_directional_measurement_before_publication(tmp_path, monkeypatch, field):
    factory, plan, *_ = configured(tmp_path, monkeypatch)
    packet, r, policy, provider, parents = registration_fixture(tmp_path, monkeypatch)
    with factory() as run:
        run.phase('qualification', plan, digest(plan), tmp_path, tmp_path/'qualification')
        run.qualified = {k: parents[k] for k in ('pilot_records', 'pilot_replays')}
        run.initial = {'record': parents['initial_record'], 'verification': parents['initial_verification']}
        r['forecast_input']['phases']['wikipedia'][field] += 1
        with pytest.raises(EvidenceError, match='forecast'):
            run.register(r, parents['source'], packet/'source-statement.sigstore.json', policy, parents['prepared'], tmp_path)
    assert provider.commits == 0
