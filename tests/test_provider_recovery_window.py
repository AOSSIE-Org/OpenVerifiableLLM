"""Bounded outage recovery through real guard loops; no provider requests."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))

from ovl_pipeline.canonical import EvidenceError, read_json
from ovl_pipeline.supervision import Journal
from probe_provider_deadline import ProviderFailure, transient_read_grace
from test_external_watchdog import Fake, NOW, intent
from test_rental_controller import RentalFake


def guard(tmp_path, kind):
    if kind == 'controller':
        fake = RentalFake(tmp_path)
        return fake, fake.directory, fake.run, fake.i['plan']['provider_terminate_epoch']
    fake = Fake(intent())
    path = tmp_path / 'watchdog'
    return fake, path, lambda: fake.run(path), fake.i['plan']['external_terminate_epoch']


@pytest.mark.parametrize('kind', ['controller', 'watchdog'])
def test_fifty_second_transient_gap_recovers_without_early_termination(tmp_path, kind):
    fake, path, run, deadline = guard(tmp_path, kind)
    original = fake.account
    failed = False

    def account():
        nonlocal failed
        if fake.alive and fake.now >= NOW + 10 and not failed:
            failed = True
            # Reproduce the recorded 50-second gap since the last good read.
            fake.sleep(40)
            raise ProviderFailure('transport', transient=True)
        return original()

    fake.account = account
    run()
    assert failed
    first = next(c[2] for c in fake.calls if c[0] == 'terminate')
    if kind == 'watchdog':
        assert first == deadline
    else:
        # The existing controller polls graceful-stop completion every 10s;
        # recovery shifts that cadence but cannot extend the external deadline.
        assert deadline <= first <= deadline + 10
        assert first <= fake.i['plan']['external_terminate_epoch']
    assert read_json(path / 'result.json')['complete']
    with Journal(path).lease() as journal:
        retries = [e['body'] for e in journal.events
                   if e['body'].get('action') == 'bounded-read-retry']
    assert len(retries) == 1
    assert retries[0]['provider_observed_epoch'] == NOW
    if kind == 'controller':
        assert fake.writes == 1


@pytest.mark.parametrize('kind', ['controller', 'watchdog'])
def test_repeated_failures_do_not_renew_window_or_fabricate_provider_reads(tmp_path, kind):
    fake, path, run, _ = guard(tmp_path, kind)
    original = fake.account

    def account():
        if fake.alive and fake.now >= NOW + 10:
            fake.sleep(18)
            raise ProviderFailure('http', status=503, transient=True)
        return original()

    fake.account = account
    run()
    first = next(c[2] for c in fake.calls if c[0] == 'terminate')
    assert NOW + 50 < first <= NOW + 120
    with Journal(path).lease() as journal:
        retries = [e['body'] for e in journal.events
                   if e['body'].get('action') == 'bounded-read-retry']
    assert len(retries) >= 2
    assert {r['provider_observed_epoch'] for r in retries} == {NOW}


@pytest.mark.parametrize('kind', ['controller', 'watchdog'])
@pytest.mark.parametrize('error', [
    ProviderFailure('http', status=401, transient=True),
    ProviderFailure('invalid-response-Refused', transient=True),
    EvidenceError('identity or digest mismatch'),
])
def test_integrity_auth_and_misclassified_errors_get_no_grace(tmp_path, kind, error):
    fake, _, run, _ = guard(tmp_path, kind)
    original = fake.account

    def account():
        if fake.alive and fake.now >= NOW + 10:
            raise error
        return original()

    fake.account = account
    run()
    assert next(c[2] for c in fake.calls if c[0] == 'terminate') == NOW + 10


@pytest.mark.parametrize('kind', ['controller', 'watchdog'])
def test_recovery_cannot_cross_existing_stop_limit(tmp_path, kind):
    fake, _, run, deadline = guard(tmp_path, kind)
    limit = fake.i['plan']['request_checkpoint_epoch'] if kind == 'controller' else deadline
    original = fake.account

    def account():
        if fake.alive and fake.now >= limit - 40:
            fake.sleep(18)
            raise ProviderFailure('transport', transient=True)
        return original()

    fake.account = account
    run()
    first = next(c[2] for c in fake.calls if c[0] == 'terminate')
    assert limit - 25 <= first <= limit


@pytest.mark.parametrize('last,mono', [(None, None), (0, None), (None, 0)])
def test_no_new_window_without_a_successful_read_in_this_process(last, mono):
    assert not transient_read_grace(ProviderFailure('transport', transient=True),
                                   50, last, 50, mono, 1000, False)
