"""Actual local safe-state delivery with injected elapsed time, not bandwidth claims."""
from datetime import datetime, timezone
import copy
import time

import pytest

from ovl_pipeline.canonical import EvidenceError, digest, read_json
from ovl_pipeline.supervision import Journal, observe, rental_plan
from workload_health import Health
from test_pilot_checkpoint_delivery import selected, job_for, request_for, hook_for
from test_pod_transfer import setup
from test_workload_stage import intent


def scenario(tmp_path, allowance):
    t, remote, calls, _ = setup(tmp_path)
    s = selected(t);s['copy_timeout_seconds'] = allowance
    job = job_for(t, s);job['deadline_epoch'] = int(time.time()) + 2700
    job['argv'][job['argv'].index('--delivery-deadline')+1] = str(job['deadline_epoch'])
    q = request_for(remote, s, job)
    w = intent();w['plan'] = rental_plan({**w['plan']['input'], 'maximum_seconds': 7200})
    w['payload']['terminateAfter'] = datetime.fromtimestamp(w['plan']['provider_terminate_epoch'], timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    start = w['plan']['input']['now_epoch'];clock = [start]
    q['copy_deadline_epoch'] = start + allowance
    t.wall = lambda: clock[0];t.monotonic = lambda: clock[0]
    def health(journal):
        return Health(journal, w, t.profile['pod_id'], wall=lambda: clock[0],
                      clock=lambda: {'boot_id': 'synthetic-clock', 'boottime_ms': (clock[0]-start+1)*1000})
    return t, remote, s, job, q, clock, health


@pytest.mark.parametrize('allowance,passes', [(660, False), (1200, True)])
def test_complete_bytes_checks_and_ack_obey_total_elapsed_window(tmp_path, allowance, passes):
    t, remote, s, job, q, clock, make_health = scenario(tmp_path, allowance)
    original = t.get;deadlines = []
    def get(name, *args, **kwargs):
        deadlines.append(args[2]);result = original(name, *args, **kwargs)
        if name.endswith('state.safetensors'):clock[0] += 850
        return result
    t.get = get
    with Journal(tmp_path/'journal').lease() as journal:
        h = make_health(journal);hook = hook_for(t, h, job, s, tmp_path)
        if passes:
            assert hook.observe({hook.marker: q})
            assert len(h.exports) == 1 and hook.index == 1
            assert read_json(remote/'record/delivery/ack-00000.json')['request_sha256'] == digest(q)
        else:
            with pytest.raises(EvidenceError):hook.observe({hook.marker: q})
            assert not h.exports and hook.index == 0
            assert not (remote/'record/delivery/ack-00000.json').exists()
        assert set(deadlines) == {q['copy_deadline_epoch']}
        assert read_json(tmp_path/'retention/checkpoint-00000/request.json') == q


@pytest.mark.parametrize('phase', ['safe-state', 'ack'])
def test_expiry_after_copy_never_authorizes_worker(tmp_path, phase):
    t, remote, s, job, q, clock, make_health = scenario(tmp_path, 1200)
    with Journal(tmp_path/'journal').lease() as journal:
        h = make_health(journal);hook = hook_for(t, h, job, s, tmp_path)
        if phase == 'safe-state':
            original = hook._check_retained
            def check(*args):
                value = original(*args);clock[0] = q['copy_deadline_epoch'];return value
            hook._check_retained = check
        else:
            original = t.put
            def put(*args, **kwargs):
                clock[0] = q['copy_deadline_epoch'];return original(*args, **kwargs)
            t.put = put
        with pytest.raises(EvidenceError):hook.observe({hook.marker: q})
        assert hook.index == 0 and not (remote/'record/delivery/ack-00000.json').exists()
        # A completed durable copy legitimately earns credit before a later ACK failure.
        assert len(h.exports) == (1 if phase == 'ack' else 0)


@pytest.mark.parametrize('elapsed', [250, 1200])
def test_reconstructed_health_and_copy_keep_original_deadline(tmp_path, elapsed):
    from pod_transfer import TransientTransportError
    t, remote, s, job, q, clock, make_health = scenario(tmp_path, 1200)
    original = t.get
    def fail(*args, **kwargs):
        error = TransientTransportError('synthetic zero-byte read interruption')
        error.transfer_counts = {'bytes_sent': 0, 'bytes_received': 0};raise error
    with Journal(tmp_path/'journal').lease() as journal:
        h = make_health(journal);hook = hook_for(t, h, job, s, tmp_path);t.get = fail
        with pytest.raises(EvidenceError):hook.observe({hook.marker: q})
        prior = (h.progress, h.exported, copy.deepcopy(h.jobs))
    clock[0] += elapsed;t.get = original
    with Journal(tmp_path/'journal').lease() as journal:
        h = make_health(journal);assert (h.progress, h.exported, h.jobs) == prior
        hook = hook_for(t, h, job, s, tmp_path)
        changed = copy.deepcopy(q);changed['copy_deadline_epoch'] += 1
        with pytest.raises(EvidenceError):hook.observe({hook.marker: changed})
        if elapsed == 1200:
            with pytest.raises(EvidenceError):hook.observe({hook.marker: q})
            assert not h.exports and not (remote/'record/delivery/ack-00000.json').exists()
        else:
            assert hook.observe({hook.marker: q})
            receipt = read_json(tmp_path/'retention/checkpoint-00000/snapshot-001/export.json')
            assert receipt['result'] == 'PASS'
        assert read_json(tmp_path/'retention/checkpoint-00000/request.json') == q


def test_long_phase_progress_cannot_replace_durable_export_health():
    from test_supervision import plan_input, observation
    p = rental_plan({**plan_input(), 'maximum_seconds': 7200})
    v = observation(p);start = p['input']['now_epoch']
    # Progress during a long transfer cannot renew the last completed export.
    for elapsed, expected in [(1700, 'CONTINUE'), (1801, 'CHECKPOINT_AND_STOP')]:
        v.update(now_epoch=start+elapsed, observed_epoch=start+elapsed,
                 progress_epoch=start+elapsed, last_checkpoint_epoch=start)
        decision = observe(p, v);assert decision['action'] == expected
        if elapsed > 1800:assert 'missing-recent-durable-checkpoint' in decision['reasons']
    # An actual intermediate export allows a longer selected phase under the same plan.
    v.update(now_epoch=start+2700, observed_epoch=start+2700,
             progress_epoch=start+2700, last_checkpoint_epoch=start+1500)
    assert observe(p, v)['action'] == 'CONTINUE'
