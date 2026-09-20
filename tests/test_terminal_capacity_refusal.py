"""Synthetic rejection recovery, never a real service/provider or budget mutation."""
import copy
import hashlib
from pathlib import Path

import pytest

import recover_capacity_refusal as m
from ovl_pipeline.canonical import EvidenceError, canonical, digest, write_json, read_json
from ovl_pipeline.supervision import Journal, ControllerBusy
from rental_safety import account_lease
from test_reconcile_capacity_rejection import fixture


def chain(events, epoch):
    result = []
    for body in events:
        result.append({'schema': 'ovl.controller-event.v1', 'sequence': len(result),
            'previous': digest(result[-1]) if result else digest({'schema': 'ovl.controller-journal.v1'}),
            'observed_epoch': epoch, **copy.deepcopy(body)})
    return result


def observations(account, times):
    return [{'schema': 'ovl.empty-account-identity-observation.v1', 'account_identity_sha256': account,
        'response_sha256': 'a' * 64, 'http_clock': {'server_epoch': t, 'request_started_epoch': t, 'request_completed_epoch': t},
        'observed_epoch': t, 'pods': [], 'volume_ids': [], 'autopay': False, 'account_hourly_usd': '0'} for t in times]


def ready():
    args = fixture()
    rental = args[0]
    watch = rental['watchdog_intent']
    now = args[-1]
    fence = {'schema': 'ovl.one-shot-creation-fence.v1', 'intent_sha256': digest(rental), 'attempt_id': rental['payload']['name']}
    args[5][1]['body']['fence_sha256'] = digest(fence)
    args[5] = chain(args[5], watch['plan']['input']['now_epoch'])
    receipt = m.verify_rejection(*args)
    watchdog = chain([{'kind': 'creation-intent', 'body': watch}], watch['plan']['input']['now_epoch'])
    current = now + 30
    heartbeat = {**args[7], 'observed_epoch': current}
    before = {'controller': args[5], 'watchdog': watchdog, 'fence': fence,
              'observations': observations(receipt['account_identity_sha256'], [now + 10, current]),
              'heartbeat': heartbeat, 'now': current}
    preparation = m.prepare(args, receipt, **before)
    inputs = {'original_args': args, 'receipt': receipt, 'before': before, 'preparation': preparation,
        'controller': before['controller'], 'watchdog': watchdog, 'fence': fence,
        'observations': observations(receipt['account_identity_sha256'], [current + 2, current + 22]),
        'service_states': {'controller': 'inactive', 'watchdog': 'inactive'},
        'stopped_epoch': current + 1, 'now': current + 22}
    return inputs


def test_terminal_refusal_closes_before_unused_deadline_without_budget_mutation():
    inputs = ready()
    result = m.finalize(**inputs)
    assert result['confirmed_absent_epoch'] < result['original_external_deadline_epoch']
    assert result['provider_final_settlement'] == 'NOT_ASSERTED'
    assert result['budget_release'].startswith('NOT_APPLIED')
    assert result['creation_fence'] == 'PRESERVE_NEVER_REISSUE_ORIGINAL_REQUEST'


@pytest.mark.parametrize('damage', ['wrong-account', 'pods', 'volume', 'rate', 'autopay', 'stale',
    'unseparated', 'request-overlap', 'before-stop', 'clock', 'fence', 'controller-running',
    'watchdog-running', 'controller-prefix', 'watchdog-prefix', 'pod-seen', 'historic-rate',
    'truncated-history', 'changed-preparation', 'changed-original-response', 'duplicate-fence'])
def test_unsafe_or_ambiguous_evidence_never_closes(damage):
    data = copy.deepcopy(ready())
    observation = data['observations'][1]
    if damage == 'wrong-account': observation['account_identity_sha256'] = 'b' * 64
    elif damage == 'pods': observation['pods'] = [{'id': 'unexpected'}]
    elif damage == 'volume': observation['volume_ids'] = ['unexpected']
    elif damage == 'rate': observation['account_hourly_usd'] = '0.01'
    elif damage == 'autopay': observation['autopay'] = True
    elif damage == 'stale': data['now'] += 31
    elif damage == 'unseparated': data['observations'][0] = copy.deepcopy(observation)
    elif damage == 'request-overlap': observation['http_clock']['request_started_epoch'] -= 10
    elif damage == 'before-stop': data['stopped_epoch'] = observation['observed_epoch']
    elif damage == 'clock': observation['http_clock']['server_epoch'] += 20
    elif damage == 'fence': data['fence']['intent_sha256'] = 'a' * 64
    elif damage == 'controller-running': data['service_states']['controller'] = 'active'
    elif damage == 'watchdog-running': data['service_states']['watchdog'] = 'active'
    elif damage == 'controller-prefix': data['controller'][0]['previous'] = 'a' * 64
    elif damage == 'watchdog-prefix': data['watchdog'][0]['previous'] = 'a' * 64
    elif damage in ('pod-seen', 'historic-rate', 'duplicate-fence'):
        previous = data['watchdog'] if damage != 'duplicate-fence' else data['controller']
        event = {'kind': 'creation-observed', 'body': {'id': 'once-existed'}}
        if damage == 'historic-rate': event = {'kind': 'provider-observation', 'body': {'pods': [], 'volume_ids': [], 'autopay': False, 'account_hourly_usd': '0.01'}}
        if damage == 'duplicate-fence': event = {'kind': 'decision', 'body': copy.deepcopy(data['controller'][1]['body'])}
        previous.append({'schema': 'ovl.controller-event.v1', 'sequence': len(previous),
                         'previous': digest(previous[-1]), 'observed_epoch': data['now'], **event})
    elif damage == 'truncated-history': data['controller'] = data['controller'][:-1]
    elif damage == 'changed-preparation': data['preparation']['observations_sha256'] = 'a' * 64
    else: data['original_args'][4] += b' '
    with pytest.raises(EvidenceError): m.finalize(**data)


def test_ambiguous_refusal_cannot_be_upgraded_by_new_policy():
    data = ready()
    data['original_args'][2]['retained_read_complete'] = False
    with pytest.raises(EvidenceError): m.finalize(**data)


def test_budget_release_is_exact_once_and_preserves_actual_spend_and_other_reserves():
    data = ready(); closure = m.finalize(**data)
    budget = {'current_rental_intent_sha256': closure['rental_intent_sha256'],
              'current_rental_projected_maximum_usd': closure['eligible_unused_creation_allowance_usd'],
              'actual_project_spend': '5.3', 'prior_unsettled_reservation_usd': '13.2', 'protected_reserve_usd': '10'}
    changed, releases = m.release_budget(budget, {}, closure, canonical(closure), closure_inputs=data)
    assert changed == {**budget, 'current_rental_projected_maximum_usd': '0'}
    assert m.release_budget(changed, releases, closure, canonical(closure), closure_inputs=data) == (changed, releases)
    with pytest.raises(EvidenceError): m.release_budget(budget, {}, closure, b'wrong', closure_inputs=data)
    with pytest.raises(EvidenceError): m.release_budget({**budget, 'current_rental_intent_sha256': 'b' * 64}, {}, closure, canonical(closure), closure_inputs=data)
    with pytest.raises(EvidenceError): m.release_budget({**budget, 'current_rental_projected_maximum_usd': '0'}, {}, closure, canonical(closure), closure_inputs=data)
    with pytest.raises(EvidenceError, match='partial prior accounting release'):
        m.release_budget(budget, releases, closure, canonical(closure), closure_inputs=data)
    with pytest.raises(EvidenceError, match='another rental'):
        m.release_budget({**changed, 'current_rental_intent_sha256': 'b' * 64}, releases,
                         closure, canonical(closure), closure_inputs=data)


class FakeGuards:
    def __init__(self):
        self.units = {'controller': 'ovllm-synthetic-controller.service', 'watchdog': 'ovllm-synthetic-watchdog.service'}
        self.hashes = {'controller': 'a' * 64, 'watchdog': 'b' * 64}
        self.states = {'controller': 'active', 'watchdog': 'active'}
        self.actions = []
    def state(self, role): return self.states[role]
    def stop(self, role): self.actions.append(('stop', role)); self.states[role] = 'inactive'
    def restore_watchdog(self): self.actions.append(('restore', 'watchdog')); self.states['watchdog'] = 'active'


def setup_disk(tmp_path, monkeypatch):
    data = ready(); args = data['original_args']; rental = args[0]
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    attempt = tmp_path / 'attempt'; attempt.mkdir()
    for name in ['capacity-reconciliation', 'provider-diagnostics', 'controller', 'watchdog']: (attempt / name).mkdir()
    write_json(attempt / 'rental-intent.json', rental)
    write_json(attempt / 'provider-diagnostics/creation-response.json', args[2])
    prior = attempt / 'capacity-reconciliation'
    for name, value in [('verification.json', data['receipt']), ('journal-prefix.json', args[5]),
                        ('account-0.json', args[6][0]), ('account-1.json', args[6][1]), ('watchdog-heartbeat.json', args[7])]: write_json(prior / name, value)
    private = tmp_path / '.local/share/openverifiablellm/provider-responses' / rental['payload']['name']; private.mkdir(parents=True)
    write_json(private / 'creation-request.json', args[3]); (private / 'creation-response.bin').write_bytes(args[4])
    for role in ['controller', 'watchdog']:
        for index, event in enumerate(data[role]): write_json(attempt / role / f'event-{index:08d}.json', event)
    write_json(attempt / 'watchdog/heartbeat.json', data['before']['heartbeat'])
    fences = tmp_path / 'fences'; fences.mkdir(mode=0o700)
    write_json(fences / (rental['payload']['name'] + '.json'), data['fence'])
    return data, attempt, fences


def test_execute_and_resume_preserve_journals_fence_and_no_double_actions(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    batches = iter([data['before']['observations'], data['observations']])
    times = iter([data['before']['now'], data['before']['now'], data['stopped_epoch'], data['now']])
    output = tmp_path / 'recovery'
    result = m.recover(attempt, digest(data['original_args'][0]), output, guards, execute=True,
                       collect=lambda: next(batches), wall=lambda: next(times), fences=fences)
    assert result == m.finalize(**data)
    assert guards.actions == [('stop', 'controller'), ('stop', 'watchdog')]
    assert Journal(attempt / 'controller')._read() == data['controller']
    assert Journal(attempt / 'watchdog')._read() == data['watchdog']
    result2 = m.recover(attempt, digest(data['original_args'][0]), output, guards, execute=True,
                        collect=lambda: pytest.fail('must not duplicate completed recovery'), fences=fences)
    assert result2 == result and len(guards.actions) == 2


def test_failure_after_stop_restores_watchdog_and_never_produces_closure(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    bad = copy.deepcopy(data['observations']); bad[1]['pods'] = [{'id': 'late'}]
    batches = iter([data['before']['observations'], bad])
    times = iter([data['before']['now'], data['before']['now'], data['stopped_epoch'], data['now']])
    output = tmp_path / 'recovery'
    with pytest.raises(EvidenceError):
        m.recover(attempt, digest(data['original_args'][0]), output, guards, execute=True,
                  collect=lambda: next(batches), wall=lambda: next(times), fences=fences)
    assert guards.states['watchdog'] == 'active'
    assert not (output / 'closure.json').exists()


def test_competing_account_owner_cannot_be_displaced(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    with account_lease(fences), pytest.raises(ControllerBusy):
        m.recover(attempt, digest(data['original_args'][0]), tmp_path / 'recovery', guards, execute=True,
                  collect=lambda: data['before']['observations'], wall=lambda: data['before']['now'], fences=fences)
    assert ('stop', 'watchdog') not in guards.actions
    assert guards.states['watchdog'] == 'active'


def test_supervisor_restores_watchdog_after_unsealed_interruption(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    output = tmp_path / 'recovery'; output.mkdir(mode=0o700)
    guards.states['watchdog'] = 'inactive'
    with pytest.raises(EvidenceError):
        m.restore_unless_closed(attempt, digest(data['original_args'][0]), output, guards, fences=fences)
    assert guards.states['watchdog'] == 'active'


def test_durable_teardown_recovers_missing_final_file_and_supervisor_checks_evidence(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    batches = iter([data['before']['observations'], data['observations']])
    times = iter([data['before']['now'], data['before']['now'], data['stopped_epoch'], data['now']])
    output = tmp_path / 'recovery'
    result = m.recover(attempt, digest(data['original_args'][0]), output, guards, execute=True,
        collect=lambda: next(batches), wall=lambda: next(times), fences=fences)
    (output / 'closure.json').unlink()
    assert m.restore_unless_closed(attempt, digest(data['original_args'][0]), output, guards, fences=fences) == result
    assert guards.states['watchdog'] == 'inactive'
    checkpoint = [e for e in Journal(output / 'recovery')._read() if e['kind'] == 'checkpoint'][-1]
    snapshot = output / (checkpoint['body']['snapshot_sha256'] + '.json')
    changed = read_json(snapshot); changed['services']['watchdog'] = 'active'; write_json(snapshot, changed)
    with pytest.raises(EvidenceError):
        m.restore_unless_closed(attempt, digest(data['original_args'][0]), output, guards, fences=fences)
    assert guards.states['watchdog'] == 'active'


def test_restore_can_restart_failed_original_unit_but_rejects_changed_unit(tmp_path, monkeypatch):
    for role in ('controller', 'watchdog'):
        (tmp_path / (role + '.service')).write_text('[Service]\nExecStart=/original-' + role + '\n')
    pins = {role: hashlib.sha256((tmp_path / (role + '.service')).read_bytes()).hexdigest()
            for role in ('controller', 'watchdog')}
    guards = m.LocalGuards(tmp_path, 'ovllm-synthetic-controller.service', 'ovllm-synthetic-watchdog.service', pins)
    output = 'FragmentPath=' + str(tmp_path / 'watchdog.service') + '\nActiveState=failed\nMainPID=0\nLoadState=loaded\nTransient=no\nDropInPaths=\nNeedDaemonReload=no\n'
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *args, **kwargs: output)
    calls = []
    monkeypatch.setattr(m.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    guards.restore_watchdog()
    assert calls == [['systemctl', '--user', 'start', 'ovllm-synthetic-watchdog.service']]
    (tmp_path / 'watchdog.service').write_text('changed')
    with pytest.raises(EvidenceError): guards.restore_watchdog()
    assert len(calls) == 1
    with pytest.raises(EvidenceError, match='caller pin'):
        m.LocalGuards(tmp_path, 'ovllm-synthetic-controller.service', 'ovllm-synthetic-watchdog.service', pins)


@pytest.mark.parametrize('changed', ['DropInPaths=/synthetic/override.conf', 'NeedDaemonReload=yes', 'LoadState=error', 'Transient=yes'])
def test_effective_unit_overrides_and_stale_loaded_config_fail(tmp_path, monkeypatch, changed):
    for role in ('controller', 'watchdog'):
        (tmp_path / (role + '.service')).write_text('[Service]\nExecStart=/synthetic\n')
    pins = {role: hashlib.sha256((tmp_path / (role + '.service')).read_bytes()).hexdigest()
            for role in ('controller', 'watchdog')}
    guards = m.LocalGuards(tmp_path, 'ovllm-synthetic-controller.service', 'ovllm-synthetic-watchdog.service', pins)
    values = {'FragmentPath': str(tmp_path / 'watchdog.service'), 'ActiveState': 'inactive', 'MainPID': '0',
              'LoadState': 'loaded', 'Transient': 'no', 'DropInPaths': '', 'NeedDaemonReload': 'no'}
    key, value = changed.split('=', 1); values[key] = value
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: '\n'.join(k + '=' + v for k, v in values.items()))
    with pytest.raises(EvidenceError, match='effective configuration'): guards.state('watchdog')


def test_competing_cleanup_never_restarts_guard_owned_by_live_recovery(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    output = tmp_path / 'recovery'; output.mkdir(mode=0o700)
    guards.states['watchdog'] = 'inactive'
    with Journal(attempt / 'terminal-recovery-operation').lease(), pytest.raises(ControllerBusy):
        m.restore_unless_closed(attempt, digest(data['original_args'][0]), output, guards, fences=fences)
    assert guards.actions == [] and guards.states['watchdog'] == 'inactive'


@pytest.mark.parametrize('damage', ['symlink-ancestor', 'shared-directory'])
def test_recovery_output_requires_private_unsymlinked_directory(tmp_path, monkeypatch, damage):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    if damage == 'symlink-ancestor':
        (tmp_path / 'alias').symlink_to(tmp_path, target_is_directory=True)
        output = tmp_path / 'alias/recovery'
    else:
        output = tmp_path / 'recovery'; output.mkdir(mode=0o755); output.chmod(0o755)
    with pytest.raises(EvidenceError):
        m.recover(attempt, digest(data['original_args'][0]), output, guards,
                  collect=lambda: pytest.fail('must reject before account read'), fences=fences)
    assert guards.actions == []


def test_final_write_failure_cannot_reuse_sealed_closure_with_restarted_guard(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    batches = iter([data['before']['observations'], data['observations']])
    times = iter([data['before']['now'], data['before']['now'], data['stopped_epoch'], data['now']])
    output = tmp_path / 'recovery'; original = m.write_json
    def fail_final(path, value):
        if path.name == 'closure.json': raise OSError('synthetic final-file failure')
        return original(path, value)
    monkeypatch.setattr(m, 'write_json', fail_final)
    with pytest.raises(OSError):
        m.recover(attempt, digest(data['original_args'][0]), output, guards, execute=True,
                  collect=lambda: next(batches), wall=lambda: next(times), fences=fences)
    assert guards.states['watchdog'] == 'active'
    monkeypatch.setattr(m, 'write_json', original)
    with pytest.raises(EvidenceError, match='guard resumed after closure'):
        m.restore_unless_closed(attempt, digest(data['original_args'][0]), output, guards, fences=fences)
    assert guards.states['watchdog'] == 'active' and not (output / 'closure.json').exists()


def test_sealed_closure_rejects_later_guard_history(tmp_path, monkeypatch):
    data, attempt, fences = setup_disk(tmp_path, monkeypatch); guards = FakeGuards()
    batches = iter([data['before']['observations'], data['observations']])
    times = iter([data['before']['now'], data['before']['now'], data['stopped_epoch'], data['now']])
    output = tmp_path / 'recovery'
    m.recover(attempt, digest(data['original_args'][0]), output, guards, execute=True,
              collect=lambda: next(batches), wall=lambda: next(times), fences=fences)
    with Journal(attempt / 'watchdog').lease() as journal:
        journal.append('provider-observation', {'pods': [{'id': 'synthetic-late-resource'}]})
    with pytest.raises(EvidenceError, match='history advanced after closure'):
        m.restore_unless_closed(attempt, digest(data['original_args'][0]), output, guards, fences=fences)
    assert guards.states['watchdog'] == 'active'
