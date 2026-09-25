"""Independent deadline guard with immutable identity and no creation authority.

Adapters provide authenticated normalized inventory and idempotent termination.
Run in a separately supervised process before allowing a rental mutation. An
empty pre-creation listing does not disarm the guard. This module never trains.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import time

from .canonical import EvidenceError, digest, read_json, write_json
from .lifecycle import Pending, RetryableRead, exclusive
from .lifecycle_creation import update as creation_update


def validate_intent(intent, expected):
    if digest(intent) != expected or set(intent) != {
            "schema", "identity", "created_not_before", "terminate_at", "grace_seconds",
            "hourly_usd", "rental_ceiling_usd", "prior_upper_usd", "stop_usd", "cap_usd"}:
        raise EvidenceError("guard intent differs from frozen pin")
    if intent["schema"] != "ovl.lifecycle-guard.v1":
        raise EvidenceError("unsupported guard intent")
    identity = intent["identity"]
    if (type(identity) is not dict or set(identity) != {"name", "gpu", "gpu_count", "cloud"}
            or not isinstance(identity["name"], str) or not identity["name"]
            or identity["gpu"] != "NVIDIA GeForce RTX 5090" or identity["gpu_count"] != 1
            or identity["cloud"] != "SECURE"):
        raise EvidenceError("guard requires the authorized single secure GPU identity")
    for name in ("created_not_before", "terminate_at", "grace_seconds"):
        if type(intent[name]) is not int or intent[name] < 0:
            raise EvidenceError("invalid frozen deadline")
    if intent["grace_seconds"] != 120 or intent["terminate_at"] <= intent["created_not_before"]:
        raise EvidenceError("invalid lifetime or grace")
    values = {}
    for name in ("hourly_usd", "rental_ceiling_usd", "prior_upper_usd", "stop_usd", "cap_usd"):
        if type(intent[name]) is not str:
            raise EvidenceError("decimal monetary strings required")
        value = Decimal(intent[name])
        if not value.is_finite() or value < 0:
            raise EvidenceError("invalid budget")
        values[name] = value
    duration = intent["terminate_at"] + intent["grace_seconds"] - intent["created_not_before"]
    maximum = values["hourly_usd"] * duration / Decimal(3600)
    if (values["hourly_usd"] <= 0 or maximum > values["rental_ceiling_usd"]
            or values["stop_usd"] != 120 or values["cap_usd"] != 130
            or values["prior_upper_usd"] + values["rental_ceiling_usd"] > values["stop_usd"]):
        raise EvidenceError("guard lifetime exceeds reconciled allowance")


def supervise(intent, expected, directory, provider, *, clock=time.time, sleep=time.sleep,
              monotonic=time.monotonic, interval=5, observation_window=120, event=lambda _: None):
    """Provider: list(deadline=monotonic_limit); terminate(id, deadline=limit).

    Required resource fields are id, name, gpu, gpu_count, cloud, created_at.
    list() must be a complete fresh authenticated inventory, or raise. Permission,
    authentication, malformed evidence and identity failures are never transient.
    Calls must be cancellable at the supplied total deadline, including pages.
    """
    validate_intent(intent, expected)
    if not 0 < interval <= 30 or not 0 < observation_window <= 120:
        raise EvidenceError("invalid guard observation bounds")
    directory = Path(directory)
    wall_start, monotonic_start = clock(), monotonic()
    wall = clock
    # A backward wall-clock adjustment cannot buy additional rental lifetime.
    clock = lambda: max(wall(), wall_start + monotonic()-monotonic_start)
    with exclusive(directory):
        path = directory / "guard.json"
        state = {"intent_sha256": expected, "resource_id": None, "status": "ARMED", "last_observed": None,
                 "last_observed_monotonic_ms": None,
                 "provisioning_match": None, "resource_seen": False, "deletion_confirmed": False,
                 "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 "stop_monotonic_ms": int((monotonic_start+intent['terminate_at']-wall_start)*1000)}
        current_boot = state['boot_id']
        if path.exists():
            state = read_json(path)
            if set(state) != {"intent_sha256", "resource_id", "status", "last_observed", "last_observed_monotonic_ms", "provisioning_match", "resource_seen", "deletion_confirmed", "boot_id", "stop_monotonic_ms"} or state["intent_sha256"] != expected:
                raise EvidenceError("guard recovery identity mismatch")
            if type(state['stop_monotonic_ms']) is not int or type(state['boot_id']) is not str:
                raise EvidenceError('invalid persisted guard clock binding')
            if state['last_observed_monotonic_ms'] is not None and type(state['last_observed_monotonic_ms']) is not int:
                raise EvidenceError('invalid persisted provider observation clock')
        if state['boot_id'] != current_boot:
            # Elapsed lifetime across a host reboot cannot be reconstructed from
            # this boot's monotonic clock. Close admission and clean up now.
            clock = lambda: max(wall(), intent['terminate_at']+monotonic()-monotonic_start)
        else:
            clock = lambda: max(wall(), intent['terminate_at']+monotonic()-state['stop_monotonic_ms']/1000)
        journal_failed = False
        def save_safety_state():
            nonlocal journal_failed
            try:
                write_json(path, state)
            except OSError:
                journal_failed = True
                # A disk failure blocks creation; it cannot suppress an already
                # authorized emergency termination of our verified resource.
                if state["resource_id"] is None:
                    raise
        # Supervision restart does not grant a new provider observation window.
        # A reboot already forces immediate cleanup using the frozen clock.
        last_success = (state['last_observed_monotonic_ms']/1000
                        if state['last_observed_monotonic_ms'] is not None and state['boot_id']==current_boot else monotonic_start)
        if last_success>monotonic_start:last_success=monotonic_start-observation_window
        def stop_requested():
            stop=directory/'stop.json'
            if not stop.exists():return False
            if read_json(stop)!={'intent_sha256':expected,'resource_id':state['resource_id']}:
                raise EvidenceError('conflicting stop identity')
            return True
        def terminate_current():
            nonlocal termination_attempted,journal_failed
            if termination_attempted:
                return
            termination_attempted = True
            state['status'] = 'TERMINATING'
            try:creation_update(directory,expected,close=True)
            except (OSError,Pending):journal_failed=True
            save_safety_state()
            try:
                if provider.terminate(state['resource_id'], deadline=monotonic()+10) is True:
                    state['deletion_confirmed'] = True
            except RetryableRead:
                pass
            save_safety_state()
        while True:
            termination_attempted = False
            now = clock()
            # A slow inventory cannot postpone termination of an already known
            # identity. Each provider operation has a total monotonic deadline.
            if now >= intent['terminate_at'] and state['resource_id'] is not None and state['status'] != 'TERMINATING':
                terminate_current()
            try:
                gate = creation_update(directory, expected, close=now >= intent['terminate_at'])
            except (OSError, Pending):
                journal_failed = True
                gate = {'admission_closed': True, 'claim': {'status': 'unresolved', 'resource_id': None}}
            claim = gate['claim']
            if claim is not None and claim['resource_id'] is not None:
                if state['resource_id'] not in (None, claim['resource_id']):
                    raise EvidenceError('guard and creation claim identify different resources')
                state['resource_id'] = claim['resource_id']
                if now >= intent['terminate_at'] and state['status'] != 'TERMINATING':
                    terminate_current()
            if monotonic()>=last_success+observation_window:
                try:gate=creation_update(directory,expected,close=True)
                except (OSError,Pending):
                    journal_failed=True;gate={**gate,'admission_closed':True}
                state['status']='TERMINATING'
                save_safety_state()
                if state['resource_id'] is not None:terminate_current()
            if state['resource_id'] is not None and state['status']!='TERMINATING' and stop_requested():
                terminate_current()
            try:
                budget = min(10, max(0.01, intent['terminate_at']-now)) if now < intent['terminate_at'] else 10
                resources = provider.list(deadline=monotonic()+budget)
            except RetryableRead:
                now = clock()
                if (now >= intent["terminate_at"] or state['status']=='TERMINATING') and state["resource_id"] is not None:
                    try:
                        provider.terminate(state["resource_id"], deadline=monotonic()+10)
                    except RetryableRead:
                        pass
                if monotonic()>=last_success+observation_window or now>=intent['terminate_at']+120:
                    try:creation_update(directory,expected,close=True)
                    except (OSError,Pending):journal_failed=True
                    if state['resource_id'] is not None:terminate_current()
                    raise Pending("guard cannot authenticate current resources; supervised recovery required")
                next_bound = intent['terminate_at'] if now < intent['terminate_at'] else intent['terminate_at']+120
                sleep(min(interval, last_success+observation_window-monotonic(), next_bound-now))
                continue
            if type(resources) is not list:
                raise EvidenceError("incomplete provider inventory")
            matching = [r for r in resources if r.get("name") == intent["identity"]["name"]]
            if len(matching) > 1:
                raise EvidenceError("multiple resources share the original operation identity")
            if state["resource_id"] is not None:
                same_id = [r for r in resources if r.get("id") == state["resource_id"]]
                if same_id and same_id != matching:
                    raise EvidenceError("known resource identity changed")
            now = clock()
            last_success = monotonic()
            state["last_observed"] = int(now)
            state['last_observed_monotonic_ms']=int(last_success*1000)
            if matching:
                resource = matching[0]
                if (type(resource.get("id")) is not str or not resource["id"]
                        or type(resource.get("created_at")) is not int
                        or resource["created_at"] < intent["created_not_before"]
                        or (resource['created_at'] >= intent['terminate_at'] and claim is None)
                        or state["resource_id"] not in (None, resource["id"])):
                    raise EvidenceError("resource identity differs from frozen guard intent")
                if state["status"] == "CLOSED" and claim is None:
                    raise EvidenceError("closed operation unexpectedly has a live resource")
                state["resource_id"] = resource["id"]
                state['resource_seen'] = True
                state["provisioning_match"] = all(resource.get(k) == v for k, v in intent["identity"].items())
                requested = stop_requested()
                # Proactively terminate at the frozen deadline. The extra 120s
                # is a billing/observation bound, never an extended run allowance.
                if state['status']=='TERMINATING' or requested or now >= intent["terminate_at"] or not state["provisioning_match"] or journal_failed:
                    state["status"] = "TERMINATING"
                    save_safety_state()
                    event("termination-intent")
                    terminate_current()
                    event("termination-returned")
                else:
                    save_safety_state()
                    event("armed")
            elif claim is not None and claim['status'] != 'not_submitted' and not (state['resource_seen'] or state['deletion_confirmed']):
                # An in-flight POST can become visible after an empty listing.
                # Keep responsibility; no expiry or empty read proves rejection.
                state['status'] = 'RECONCILING' if now >= intent['terminate_at'] else 'ARMED'
                save_safety_state()
                event('creation-unresolved')
            elif state["resource_id"] is not None or gate['admission_closed']:
                state["status"] = "CLOSED"
                save_safety_state()
                if journal_failed:
                    raise EvidenceError("resource absence verified; guard journal write failed")
                return state
            else:
                save_safety_state()
                event("armed")
            if now >= intent["terminate_at"] + 120:
                raise Pending("original creation/resource unresolved beyond grace; supervised cleanup remains required")
            now=clock()
            next_bound=intent['terminate_at'] if now<intent['terminate_at'] else intent['terminate_at']+120
            sleep(min(interval,max(0.01,next_bound-now)))
