"""A prospective delivery allowance never renews a fixed request or job deadline."""
import pytest

from ovl_pipeline import pilot_delivery as delivery, training
from ovl_pipeline.canonical import EvidenceError, read_json, write_json
from ovl_pipeline.state import save_state
from test_pilot_checkpoint_delivery import recipe


def run_delivery(tmp_path, allowance, job_deadline, elapsed, *, corrupt=False):
    policy = {'schema': 'ovl.pilot-delivery-policy.v1', 'session': 'a' * 64,
              'mode': 'record', 'phase': 'wikipedia', 'deadline_epoch': job_deadline,
              'copy_timeout_seconds': allowance, 'maximum_checkpoint_bytes': 10 * 1024**2}
    output = tmp_path / 'record'
    model, optimizer, control = training.initialize(recipe(300))
    control.update(phase='wikipedia', pilot_cycle=0)
    checkpoint = save_state(output / 'boundary-00000', model, optimizer, control)
    clock = [100]

    def deliver(_):
        request = read_json(output / 'delivery/request.json')
        ack = delivery.acknowledgement(request, 'c' * 64)
        if corrupt:
            ack['state_root'] = 'd' * 64
        write_json(output / 'delivery/ack-00000.json', ack)
        clock[0] = 100 + elapsed

    sender = delivery.Delivery(output, policy, 'b' * 64,
                               wall=lambda: clock[0], sleep=deliver)
    return sender, output, checkpoint, control


@pytest.mark.parametrize('allowance,elapsed,passes', [
    (420, 530, False), (660, 530, True), (660, 659, True),
    (660, 660, False), (420, 420, False),
])
def test_measured_slow_copy_requires_prospective_allowance(tmp_path, allowance, elapsed, passes):
    sender, output, checkpoint, control = run_delivery(tmp_path, allowance, 2000, elapsed)
    if passes:
        sender.checkpoint('boundary-00000', checkpoint, control)
        assert sender.index == 1
    else:
        with pytest.raises(EvidenceError, match='deadline expired'):
            sender.checkpoint('boundary-00000', checkpoint, control)
        assert sender.index == 0
    request = read_json(output / 'delivery/request.json')
    assert request['copy_deadline_epoch'] == 100 + allowance
    assert request['policy']['deadline_epoch'] == 2000
    assert read_json(output / 'delivery/request-00000.json') == request


def test_larger_allowance_cannot_outlive_original_job(tmp_path):
    sender, output, checkpoint, control = run_delivery(tmp_path, 660, 500, 400)
    with pytest.raises(EvidenceError, match='deadline expired'):
        sender.checkpoint('boundary-00000', checkpoint, control)
    assert read_json(output / 'delivery/request.json')['copy_deadline_epoch'] == 500
    assert sender.index == 0


def test_longer_window_still_rejects_wrong_state_ack(tmp_path):
    sender, _, checkpoint, control = run_delivery(tmp_path, 660, 2000, 530, corrupt=True)
    with pytest.raises(EvidenceError, match='acknowledgement'):
        sender.checkpoint('boundary-00000', checkpoint, control)
    assert sender.index == 0


@pytest.mark.parametrize('allowance', [29, 661, 900, True, '660', 660.0])
def test_copy_allowance_remains_bounded_and_integral(tmp_path, allowance):
    with pytest.raises(EvidenceError):
        run_delivery(tmp_path, allowance, 2000, 530)
