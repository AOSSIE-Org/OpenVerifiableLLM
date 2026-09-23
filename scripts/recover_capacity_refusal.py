#!/usr/bin/env python3
"""Close a verified supply refusal without waiting for an unused rental lifetime.

The provider's complete authenticated refusal is trusted as a terminal outcome,
along with its identified account observations. The exact originally selected
network volume may remain within its unchanged separate reservation. This is an explicit operational
trust policy, not a provider SLA, proof of noncreation, or final billing evidence.
Timeouts and ambiguous responses retain the original deadline recovery path.

No provider mutation, fence removal, budget write or successor creation occurs.
Execute mode stops only the two exactly pinned local guard units, under exclusive
account/journal ownership, and verifies absence again before issuing eligibility.
All outputs are private; publish only a reviewed technical derivative.
"""
from contextlib import ExitStack
from decimal import Decimal
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time

from ovl_pipeline.canonical import EvidenceError, digest, read_json, require_digest, write_json
from ovl_pipeline.schema import fields, integer
from ovl_pipeline.supervision import Journal
from reconcile_capacity_rejection import capture, verify as verify_rejection, selected_baseline, identified_baseline, retained_window
from rental_safety import account_lease, creation_root
from provider_request_receipts import private_directory


def insist(condition, message):
    if not condition:
        raise EvidenceError(message)


def journal(events, intent, now):
    """Validate the complete retained chain, including events after reconciliation."""
    insist(type(events) is list and bool(events), 'complete guard journal required')
    previous = digest({'schema': 'ovl.controller-journal.v1'})
    watch = intent.get('watchdog_intent', intent)
    for i, event in enumerate(events):
        fields(event, 'schema sequence previous observed_epoch kind body', 'guard event')
        insist(event['schema'] == 'ovl.controller-event.v1' and event['sequence'] == i
                and event['previous'] == previous, 'guard journal ancestry differs')
        integer(event['observed_epoch'], 1, now, 'guard event time')
        insist(type(event['body']) is dict, 'invalid guard event body')
        previous = digest(event)
        insist(event['kind'] != 'creation-observed', 'resource was observed; normal teardown required')
        body = event['body']
        if event['kind'] == 'provider-observation':
            observation = body.get('account', body)
            selected_baseline(watch, observation)
        if event['kind'] in ('teardown', 'decision'):
            insist(body.get('pod_id') is None, 'guard attributed a resource')
    insist([e['body'] for e in events if e['kind'] == 'creation-intent'] == [intent],
            'guard intent identity differs')


def identified_absence(observations, account_identity, earliest, now, watch=None):
    insist(type(observations) is list and len(observations) == 2, 'two identified absence reads required')
    for observation in observations:
        identified_baseline(observation, watch or {}, account_identity)
        integer(observation['observed_epoch'], earliest, now, 'absence time')
        clock = observation['http_clock']
        fields(clock, 'server_epoch request_started_epoch request_completed_epoch', 'provider clock')
        for value in clock.values():
            integer(value, 1, 2**53 - 1, 'provider clock value')
        insist(earliest <= clock['request_started_epoch'] <= clock['request_completed_epoch'] <= observation['observed_epoch'],
                'absence request predates closure boundary')
        insist(clock['request_started_epoch'] - 5 <= clock['server_epoch'] <= clock['request_completed_epoch'] + 5,
                'provider clock differs')
    insist(observations[1]['http_clock']['request_started_epoch'] >= observations[0]['http_clock']['request_completed_epoch'] + 15,
            'absence requests must be separated')
    insist(0 <= now - observations[1]['observed_epoch'] <= 30, 'absence observation is stale')


def verify_history(original_args, receipt, controller, watchdog, fence, now):
    insist(verify_rejection(*original_args) == receipt, 'original refusal evidence differs')
    rental = original_args[0]
    watch = rental['watchdog_intent']
    retained_window(watch, now)
    journal(controller, rental, now)
    journal(watchdog, watch, now)
    insist(controller[:len(original_args[5])] == original_args[5], 'original creator prefix changed')
    expected = {'schema': 'ovl.one-shot-creation-fence.v1', 'intent_sha256': digest(rental),
                'attempt_id': rental['payload']['name']}
    insist(fence == expected, 'durable creation fence differs')
    fences = [e['body'] for e in controller if e['kind'] == 'decision' and e['body'].get('action') == 'CREATION_FENCED']
    insist(fences == [{'action': 'CREATION_FENCED', 'fence_sha256': digest(fence)}], 'original creation fence was changed or duplicated')
    return rental, watch


def prepare(original_args, receipt, controller, watchdog, fence, observations, heartbeat, now):
    integer(now, receipt['observed_epoch'], 2**53 - 1, 'recovery time')
    rental, watch = verify_history(original_args, receipt, controller, watchdog, fence, now)
    identified_absence(observations, receipt['account_identity_sha256'],
                       max(watch['creation_latest_epoch'] + 180, receipt['observed_epoch']), now, watch)
    fields(heartbeat, 'schema intent_sha256 plan_sha256 external_terminate_epoch observed_epoch state pod_id pid automatic_provider_termination', 'watchdog heartbeat')
    insist(heartbeat['schema'] == 'ovl.external-watchdog-heartbeat.v1'
            and heartbeat['intent_sha256'] == digest(watch)
            and heartbeat['plan_sha256'] == digest(watch['plan'])
            and heartbeat['external_terminate_epoch'] == watch['plan']['external_terminate_epoch']
            and heartbeat['state'] in ('ARMED', 'TERMINATING')
            and heartbeat['pod_id'] is None and heartbeat['automatic_provider_termination'] == 'UNVERIFIED',
            'original watchdog identity differs')
    integer(heartbeat['pid'], 1, 2**31 - 1, 'watchdog pid')
    integer(heartbeat['observed_epoch'], max(1, now - 30), now, 'fresh watchdog time')
    return {'schema': 'ovl.terminal-capacity-refusal-preparation.v1',
            'result': 'ELIGIBLE_FOR_LOCAL_GUARD_CLOSURE', 'observed_epoch': now,
            'rental_intent_sha256': digest(rental), 'rejection_receipt_sha256': digest(receipt),
            'controller_events_sha256': digest(controller), 'watchdog_events_sha256': digest(watchdog),
            'observations_sha256': digest(observations), 'heartbeat_sha256': digest(heartbeat),
            'creation_fence_sha256': digest(fence), 'budget_release': 'NOT_APPLIED',
            'trust_basis': 'COMPLETE_PROVIDER_SUPPLY_REFUSAL_AND_IDENTIFIED_ACCOUNT_READS'}


def finalize(original_args, receipt, before, preparation, controller, watchdog, fence,
             observations, service_states, stopped_epoch, now):
    insist(prepare(original_args, receipt, **before) == preparation, 'preparation evidence differs')
    rental, watch = verify_history(original_args, receipt, controller, watchdog, fence, now)
    insist(controller[:len(before['controller'])] == before['controller']
            and watchdog[:len(before['watchdog'])] == before['watchdog'], 'guard history changed during closure')
    integer(stopped_epoch, preparation['observed_epoch'], now, 'guard stop time')
    insist(service_states == {'controller': 'inactive', 'watchdog': 'inactive'}, 'original guards must be inactive')
    identified_absence(observations, receipt['account_identity_sha256'], stopped_epoch, now, watch)
    allowance = Decimal(watch['plan']['maximum_charge_micro_usd']) / 10**6
    result = {'schema': 'ovl.terminal-capacity-refusal-closure.v1', 'result': 'PASS',
            'attempt_id': rental['payload']['name'], 'rental_intent_sha256': digest(rental),
            'original_rejection_sha256': digest(receipt), 'preparation_sha256': digest(preparation),
            'controller_events_sha256': digest(controller), 'watchdog_events_sha256': digest(watchdog),
            'creation_fence_sha256': digest(fence), 'post_closure_observations_sha256': digest(observations),
            'original_external_deadline_epoch': watch['plan']['external_terminate_epoch'],
            'confirmed_absent_epoch': observations[-1]['observed_epoch'], 'guards_stopped_epoch': stopped_epoch,
            'eligible_unused_creation_allowance_usd': format(allowance, 'f'),
            'budget_release': 'NOT_APPLIED_REQUIRES_VERIFIED_PUBLIC_RECONCILIATION',
            'provider_final_settlement': 'NOT_ASSERTED', 'provider_mutation': 'NOT_RUN',
            'creation_fence': 'PRESERVE_NEVER_REISSUE_ORIGINAL_REQUEST',
            'trust_basis': 'Authenticated supply refusal treated as terminal; same-account observations trusted. Not a provider SLA or proof against hidden provider state.',
            'training_verification_credit': False}
    if watch.get('retained_volume') is not None:
        result.update(schema='ovl.terminal-capacity-refusal-closure.v2',
                      retained_volume_sha256=digest(watch['retained_volume']),
                      retained_storage_reserved_usd=watch['retained_volume']['reserved_usd'],
                      storage_action='RETAIN_UNCHANGED_NO_STORAGE_RESERVATION_RELEASE')
    return result


def release_budget(budget, releases, closure, public_bytes, *, closure_inputs):
    """Pure, at-most-once accounting transition; caller persists under its ledger lock.

    public_bytes must be obtained by the existing trusted public-download verifier.
    The complete closure is already a technical derivative with no raw account ID.
    This function changes only the matching attempt's unused current allowance.
    It neither decreases actual spend nor changes other unsettled reservations.
    """
    from ovl_pipeline.canonical import canonical
    insist(finalize(**closure_inputs) == closure, 'closure evidence differs')
    insist(public_bytes == canonical(closure), 'public reconciliation bytes differ')
    if closure.get('schema') == 'ovl.terminal-capacity-refusal-closure.v2':
        from ovl_pipeline.budget import money
        reserved = money(closure['retained_storage_reserved_usd'])
        insist(money(budget.get('retained_volume_reserved_usd')) >= reserved
                and money(budget.get('remaining_mandatory_reservation_usd')) >= reserved,
                'current retained storage reservation is not covered')
    insist(closure.get('schema') in ('ovl.terminal-capacity-refusal-closure.v1', 'ovl.terminal-capacity-refusal-closure.v2')
            and closure.get('result') == 'PASS'
            and closure.get('budget_release') == 'NOT_APPLIED_REQUIRES_VERIFIED_PUBLIC_RECONCILIATION',
            'invalid terminal refusal closure')
    key = closure['rental_intent_sha256']
    require_digest(key)
    expected = {'closure_sha256': digest(closure), 'released_usd': closure['eligible_unused_creation_allowance_usd']}
    insist(budget.get('current_rental_intent_sha256') == key, 'accounting selected another rental')
    if key in releases:
        insist(releases[key] == expected, 'conflicting prior accounting release')
        insist(Decimal(budget['current_rental_projected_maximum_usd']) == 0,
                'partial prior accounting release')
        return dict(budget), dict(releases)
    amount = Decimal(expected['released_usd'])
    insist(amount.is_finite() and amount > 0
            and Decimal(budget['current_rental_projected_maximum_usd']) == amount,
            'remaining allowance differs; no partial or duplicate release')
    changed = dict(budget)
    changed['current_rental_projected_maximum_usd'] = '0'
    recorded = dict(releases)
    recorded[key] = expected
    return changed, recorded


class LocalGuards:
    def __init__(self, attempt, controller_unit, watchdog_unit, expected_hashes):
        self.attempt = attempt
        self.units = {'controller': controller_unit, 'watchdog': watchdog_unit}
        self.hashes = {}
        for role, unit in self.units.items():
            insist(re.fullmatch(r'ovllm-[a-z0-9-]+-' + role + r'\.service', unit), 'unexpected guard unit name')
            require_digest(expected_hashes[role])
            retained = attempt / (role + '.service')
            insist(not retained.is_symlink()
                    and hashlib.sha256(retained.read_bytes()).hexdigest() == expected_hashes[role],
                    'retained guard unit differs from caller pin')
            self.hashes[role] = expected_hashes[role]

    def state(self, role, *, allow_failed=False):
        unit = self.units[role]
        data = subprocess.check_output(['systemctl', '--user', 'show', unit,
                                       '--property=FragmentPath,ActiveState,MainPID,DropInPaths,NeedDaemonReload,LoadState,Transient'], text=True, timeout=20)
        values = dict(line.split('=', 1) for line in data.splitlines())
        insist(values.get('LoadState') == 'loaded' and values.get('Transient') == 'no'
                and values.get('DropInPaths') == '' and values.get('NeedDaemonReload') == 'no',
                'guard has overrides, unloaded or changed effective configuration')
        path = Path(values['FragmentPath'])
        insist(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == self.hashes[role], 'guard unit bytes changed')
        state = values['ActiveState']
        allowed = ('active', 'inactive', 'failed') if allow_failed else ('active', 'inactive')
        insist(state in allowed and (state == 'active' or values['MainPID'] == '0'), 'guard is not stably active/inactive')
        return state

    def stop(self, role):
        self.state(role)
        subprocess.run(['systemctl', '--user', 'stop', self.units[role]], check=True, timeout=60)
        insist(self.state(role) == 'inactive', 'guard did not stop')

    def restore_watchdog(self):
        self.state('watchdog', allow_failed=True)
        subprocess.run(['systemctl', '--user', 'start', self.units['watchdog']], check=True, timeout=30)


def collect_absence(watch=None):
    first = capture(watch)
    time.sleep(16)
    return [first, capture(watch)]


def recover(attempt, expected, output, guards, *, execute=False, collect=None,
            wall=time.time, fences=None):
    output = private_directory(output)
    with Journal(attempt / 'terminal-recovery-operation').lease():
        return _recover(attempt, expected, output, guards, execute=execute,
                        collect=collect, wall=wall, fences=fences)


def _recover(attempt, expected, output, guards, *, execute=False, collect=None,
             wall=time.time, fences=None):
    rental = read_json(attempt / 'rental-intent.json')
    insist(digest(rental) == expected, 'rental differs from caller pin')
    if collect is None:collect = lambda: collect_absence(rental['watchdog_intent'])
    prior = attempt / 'capacity-reconciliation'
    receipt = read_json(prior / 'verification.json')
    private = Path.home() / '.local/share/openverifiablellm/provider-responses' / rental['payload']['name']
    args = [rental, expected, read_json(attempt / 'provider-diagnostics/creation-response.json'),
            read_json(private / 'creation-request.json'), (private / 'creation-response.bin').read_bytes(),
            read_json(prior / 'journal-prefix.json'), [read_json(prior / ('account-' + str(i) + '.json')) for i in (0, 1)],
            read_json(prior / 'watchdog-heartbeat.json'), receipt['observed_epoch']]
    root = fences or creation_root()
    fence_path = root / (rental['payload']['name'] + '.json')
    insist(not output.is_symlink(), 'recovery output symlink')
    output.mkdir(mode=0o700, parents=False, exist_ok=True)
    with Journal(output / 'recovery').lease() as recovery:
        identity = {'schema': 'ovl.terminal-refusal-recovery-intent.v1', 'rental_intent_sha256': expected,
                    'units': guards.units, 'unit_sha256': guards.hashes}
        if not recovery.events:
            recovery.append('creation-intent', identity)
        insist(recovery.events[0]['body'] == identity, 'recovery intent differs; adopt original work')
        sealed = bool(recovery.events and recovery.events[-1]['kind'] == 'teardown')
        if (output / 'closure.json').exists() or sealed:
            saved = read_json(output / 'closure.json') if (output / 'closure.json').exists() else recovery.events[-1]['body']
            insist(saved['rental_intent_sha256'] == expected, 'saved closure differs')
            insist(recovery.events[-1]['kind'] == 'teardown' and recovery.events[-1]['body'] == saved,
                    'saved closure differs from recovery journal')
            checkpoint = [e['body'] for e in recovery.events if e['kind'] == 'checkpoint'][-1]
            last = read_json(output / (checkpoint['snapshot_sha256'] + '.json'))
            insist(digest(last) == checkpoint['snapshot_sha256'], 'closure snapshot changed')
            verified = finalize(args, receipt, read_json(output / 'before.json'),
                read_json(output / 'preparation.json'), last['controller'], last['watchdog'],
                read_json(fence_path), last['observations'], last['services'], last['stopped_epoch'], last['verified_epoch'])
            insist(verified == saved, 'saved closure evidence changed')
            insist({role: guards.state(role) for role in ('controller', 'watchdog')}
                    == {'controller': 'inactive', 'watchdog': 'inactive'},
                    'guard resumed after closure; fresh supervised recovery required')
            insist(Journal(attempt / 'controller')._read() == last['controller']
                    and Journal(attempt / 'watchdog')._read() == last['watchdog'],
                    'guard history advanced after closure; fresh supervised recovery required')
            # Recover a crash between the durable teardown event and final file.
            if not (output / 'closure.json').exists():
                write_json(output / 'closure.json', saved)
            # Read-only service/history checks above; no provider calls or mutations.
            return saved
        prep_path = output / 'preparation.json'
        if not prep_path.exists():
            insist(guards.state('watchdog') == 'active', 'original watchdog must protect preparation')
            guards.state('controller')
            observations = collect()
            before = {'controller': Journal(attempt / 'controller')._read(),
                      'watchdog': Journal(attempt / 'watchdog')._read(), 'fence': read_json(fence_path),
                      'observations': observations, 'heartbeat': read_json(attempt / 'watchdog/heartbeat.json'),
                      'now': int(wall())}
            preparation = prepare(args, receipt, **before)
            write_json(output / 'before.json', before)
            write_json(prep_path, preparation)
        else:
            before = read_json(output / 'before.json')
            preparation = read_json(prep_path)
            insist(prepare(args, receipt, **before) == preparation, 'saved preparation changed')
        if not execute:
            return preparation
        # The old creator is permanently fenced; stop it before taking the host's
        # creation lock. The watchdog remains active until that ownership is held.
        guards.stop('controller')
        try:
            with account_lease(root), ExitStack() as stack:
                controller = stack.enter_context(Journal(attempt / 'controller').lease())
                verify_history(args, receipt, controller.events, Journal(attempt / 'watchdog')._read(), read_json(fence_path), int(wall()))
                guards.stop('watchdog')
                stopped = int(wall())
                watchdog = stack.enter_context(Journal(attempt / 'watchdog').lease())
                observations = collect()
                states = {role: guards.state(role) for role in ('controller', 'watchdog')}
                verified_epoch = int(wall())
                result = finalize(args, receipt, before, preparation, controller.events,
                                  watchdog.events, read_json(fence_path), observations, states, stopped, verified_epoch)
                # Store complete private verification inputs before publishing the
                # result. Original guard journals and deadlines are never edited.
                snapshot = {'observations': observations, 'services': states,
                    'controller': controller.events, 'watchdog': watchdog.events,
                    'stopped_epoch': stopped, 'verified_epoch': verified_epoch}
                snapshot_hash = digest(snapshot)
                write_json(output / (snapshot_hash + '.json'), snapshot)
                recovery.append('checkpoint', {'snapshot_sha256': snapshot_hash})
                recovery.append('teardown', result)
                write_json(output / 'closure.json', result)
                return result
        except BaseException:
            # Never leave an uncertain attempt unguarded after a failed recovery.
            guards.restore_watchdog()
            raise


def restore_unless_closed(attempt, expected, output, guards, *, fences=None):
    """Systemd ExecStopPost entry: restore the original watchdog after interruption.

    A separately supervised process invokes this even when recovery receives
    SIGKILL. Do not treat an unverified closure filename as permission to disarm.
    """
    output = private_directory(output)
    # Busy means another live recovery owns service transitions. It must never
    # trigger this process's restoration handler.
    with Journal(attempt / 'terminal-recovery-operation').lease():
        try:
            events = Journal(output / 'recovery')._read()
            insist(bool(events) and events[-1]['kind'] == 'teardown', 'closure is not durably sealed')
            result = _recover(attempt, expected, output, guards, fences=fences,
                              collect=lambda: insist(False, 'closure must not need fresh preparation'))
            insist(result.get('result') == 'PASS', 'closure verification failed')
            return result
        except BaseException:
            guards.restore_watchdog()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--expected-rental', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controller-unit', required=True)
    parser.add_argument('--watchdog-unit', required=True)
    parser.add_argument('--expected-controller-unit-sha256', required=True)
    parser.add_argument('--expected-watchdog-unit-sha256', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--restore-watchdog-unless-closed', action='store_true', help='supervisor ExecStopPost: restore original watchdog unless durable closure verifies')
    mode.add_argument('--execute', action='store_true', help='stop exact original local guards after verification; never release budget or create resources')
    args = parser.parse_args()
    os.umask(0o077)
    guards = LocalGuards(args.attempt, args.controller_unit, args.watchdog_unit,
        {'controller': args.expected_controller_unit_sha256,
         'watchdog': args.expected_watchdog_unit_sha256})
    if args.restore_watchdog_unless_closed:
        result = restore_unless_closed(args.attempt, args.expected_rental, args.output, guards)
    else:
        result = recover(args.attempt, args.expected_rental, args.output, guards, execute=args.execute)
    print({'result': result['result'], 'budget_release': result['budget_release'], 'provider_mutation': 'NOT_RUN'})


if __name__ == '__main__':
    main()
