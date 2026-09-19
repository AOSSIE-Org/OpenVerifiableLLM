"""Observation freshness must never substitute for progress or retained bytes."""
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from workload_heartbeat import Heartbeat
from ovl_pipeline.canonical import EvidenceError, read_json
from ovl_pipeline.supervision import Journal, observe, rental_plan
from test_workload_health import Clock, SELECTION, JOB, export
from test_supervision import plan_input, observation


def test_background_observation_does_not_append_progress_or_export(tmp_path, monkeypatch):
    clock = Clock()
    with Journal(tmp_path / 'journal').lease() as journal:
        health = clock.health(journal)
        health.start_job(SELECTION)
        count = len(journal.events)
        path = tmp_path / 'health.json'
        original = health.pulse
        observed = threading.Event()
        def pulse(target):
            result = original(target)
            if threading.current_thread().name == 'ovllm-health-observation':
                observed.set()
            return result
        monkeypatch.setattr(health, 'pulse', pulse)
        with Heartbeat(health, path, interval_seconds=1) as heartbeat:
            clock.advance(301)
            assert observed.wait(4)
            heartbeat.check()
            value = read_json(path)
            assert value['observed_epoch'] == clock.now
            assert value['progress_epoch'] == value['exported_checkpoint_epoch'] == clock.now - 301
            assert len(journal.events) == count
        assert not heartbeat.thread.is_alive()
        with pytest.raises(EvidenceError, match='cannot be restarted'):
            heartbeat.__enter__()


@pytest.mark.parametrize('damage', ['deadline', 'boot', 'write'])
def test_background_failure_is_propagated_without_new_clock(tmp_path, monkeypatch, damage):
    clock = Clock()
    with Journal(tmp_path / 'journal').lease() as journal:
        health = clock.health(journal)
        with pytest.raises(EvidenceError):
            with Heartbeat(health, tmp_path / 'health.json', interval_seconds=1) as heartbeat:
                if damage == 'deadline': clock.advance(1000)
                elif damage == 'boot': clock.boot = 'changed'
                else:
                    monkeypatch.setattr(health, 'pulse', lambda path: (_ for _ in ()).throw(OSError('write failure')))
                assert heartbeat.stop.wait(4)
                heartbeat.check()
        assert not heartbeat.thread.is_alive()


def test_original_operation_error_is_preserved(tmp_path):
    clock = Clock()
    with Journal(tmp_path / 'journal').lease() as journal:
        with pytest.raises(ValueError, match='original integrity failure'):
            with Heartbeat(clock.health(journal), tmp_path / 'health.json'):
                raise ValueError('original integrity failure')


def test_lost_lease_rejects_observation(tmp_path):
    clock = Clock()
    with Journal(tmp_path / 'journal').lease() as journal:
        health = clock.health(journal)
    with pytest.raises(EvidenceError, match='lease'):
        health.pulse(tmp_path / 'health.json')
    with pytest.raises(EvidenceError, match='lease'):
        Heartbeat(health, tmp_path / 'health.json').__enter__()
    assert not (tmp_path / 'health.json').exists()


def test_adopted_completion_requires_real_byte_revalidation(tmp_path):
    clock = Clock(); path = tmp_path / 'journal'
    with Journal(path).lease() as journal:
        health = clock.health(journal); health.start_job(SELECTION)
        root, files, status = export(tmp_path, health)
        health.job_exit(JOB, status); health.finish(root, files)
        assert health.pulse(tmp_path / 'complete.json')['complete']
    with Journal(path).lease() as journal:
        health = clock.health(journal)
        assert not health.pulse(tmp_path / 'adopted.json')['complete']
        assert health.write(tmp_path / 'verified.json')['complete']
    (root / 'state.safetensors').write_bytes(b'altered')
    with Journal(path).lease() as journal:
        health = clock.health(journal)
        assert not health.pulse(tmp_path / 'altered.json')['complete']
        with pytest.raises(EvidenceError): health.write(tmp_path / 'must-not-exist.json')
        assert not health._complete_verified
    assert not (tmp_path / 'must-not-exist.json').exists()


@pytest.mark.parametrize('field,age,reason', [
    ('progress_epoch', 301, 'stalled-or-future-progress'),
    ('last_checkpoint_epoch', 1801, 'missing-recent-durable-checkpoint'),
])
def test_fresh_observation_cannot_defeat_existing_age_guards(field, age, reason):
    plan = rental_plan(plan_input()); value = observation(plan)
    value[field] = value['now_epoch'] - age
    assert value['observed_epoch'] == value['now_epoch']
    result = observe(plan, value)
    assert result['action'] == 'CHECKPOINT_AND_STOP'
    assert reason in result['reasons']
