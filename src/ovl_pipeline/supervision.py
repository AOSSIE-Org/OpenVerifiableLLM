"""Durable single-controller journal and conservative per-rental deadline policy.

No API mutation is implemented here. A plan is not a verified provider guard.
Provider creation/readback, spend reconciliation and remote checkpoint integration
must be supplied by the controller before paid execution can be admitted.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import tempfile
import time

from .budget import CAP, OPERATING_LIMIT, ceil_div, money
from .canonical import EvidenceError, canonical, digest, parse_json, require_digest
from .schema import fields, integer


def rental_plan(value):
    revised = value.get("schema") == "ovl.rental-budget-input.v2"
    fields(value, "schema attempt_id now_epoch spent_usd outstanding_usd reserved_remaining_usd allowance_usd hourly_upper_usd quote_sha256 maximum_seconds checkpoint_grace_seconds billing_slack_seconds" + (" external_termination_grace_seconds authorization_sha256" if revised else ""), "rental budget input")
    if value["schema"] not in ("ovl.rental-budget-input.v1", "ovl.rental-budget-input.v2"):raise EvidenceError("unsupported rental budget input")
    fallback = 0
    if revised:
        require_digest(value['authorization_sha256'])
        integer(value['external_termination_grace_seconds'],120,120,'authorized termination grace')
        fallback = value['external_termination_grace_seconds']
    if type(value["attempt_id"]) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value["attempt_id"]):
        raise EvidenceError("invalid rental attempt")
    require_digest(value["quote_sha256"])
    now = value["now_epoch"];integer(now, 1, 2**53 - 1, "current epoch")
    maximum = value["maximum_seconds"];integer(maximum, 1, 7 * 86400, "maximum rental seconds")
    grace = value["checkpoint_grace_seconds"];integer(grace, 300, 3600, "checkpoint grace")
    slack = value["billing_slack_seconds"];integer(slack, 300, 3600, "provider billing slack")
    spent, outstanding, reserved, allowance, hourly = [money(value[k]) for k in
        ("spent_usd", "outstanding_usd", "reserved_remaining_usd", "allowance_usd", "hourly_upper_usd")]
    if hourly <= 0 or allowance <= 0:raise EvidenceError("positive all-in quote and allowance required")
    available = min(allowance, OPERATING_LIMIT - spent - outstanding - reserved)
    seconds = min(maximum, available * 3600 // hourly - slack - fallback)
    if seconds <= grace:
        raise EvidenceError("remaining funds cannot cover rental, checkpoint grace and billing slack")
    # The absolute lifetime starts before provisioning, not when training starts.
    maximum_charge = ceil_div((seconds + fallback + slack) * hourly, 3600)
    if spent + outstanding + reserved + maximum_charge > OPERATING_LIMIT:
        raise EvidenceError("rental would consume protected reserve")
    return {"schema": "ovl.rental-budget-plan.v2" if revised else "ovl.rental-budget-plan.v1", "input": value, "input_sha256": digest(value),
            "maximum_charge_micro_usd": maximum_charge,
            "request_checkpoint_epoch": now + seconds - grace,
            "provider_terminate_epoch": now + seconds,
            "billing_ceiling_epoch": now + seconds + fallback + slack,
            "protected_reserve_micro_usd": CAP - OPERATING_LIMIT,
            "provider_guard": "NOT_RUN", "execution_admission": "NOT_RUN",
            **({'external_terminate_epoch':now+seconds+fallback,
                'automatic_provider_termination':'UNVERIFIED',
                'external_watchdog':'NOT_RUN'} if revised else {})}


def observe(plan, value):
    """Decide continuation from normalized provider/controller observations.

    The caller must obtain and authenticate observations itself. A positive result
    is a policy decision only, never evidence that an API guard actually exists.
    STOP requires checkpoint/avoid idle compute; unrelated resources are untouched.
    """
    if plan != rental_plan(plan["input"]):raise EvidenceError("rental plan differs from bounded input")
    revised = plan['schema'] == 'ovl.rental-budget-plan.v2'
    guard_fields = ('terminate_after_request_epoch watchdog_observed_epoch watchdog_plan_sha256 watchdog_external_terminate_epoch watchdog_state' if revised else
                    'provider_terminate_epoch provider_guard_verified')
    fields(value, "schema now_epoch observed_epoch attributed_pod_ids active_pod_ids pod_id gpu_count hourly_usd actual_project_spend_usd outstanding_usd reserved_remaining_usd account_balance_usd progress_epoch last_checkpoint_epoch " + guard_fields, "supervisor observation")
    if value["schema"] != ('ovl.supervisor-observation.v2' if revised else 'ovl.supervisor-observation.v1'):raise EvidenceError("unsupported observation")
    for k in ("now_epoch", "observed_epoch", "progress_epoch", "last_checkpoint_epoch"):
        integer(value[k], 1, 2**53 - 1, k)
    integer(value["gpu_count"], 0, 1024, "observed GPU count")
    ids = value["attributed_pod_ids"];active = value["active_pod_ids"]
    for names in (ids, active):
        if (type(names) is not list or len(names) != len(set(names))
                or any(type(n) is not str or not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", n) for n in names)):
            raise EvidenceError("invalid resource inventory")
    if value["pod_id"] not in ids:
        raise EvidenceError("resource lacks project creation attribution; no mutation authority")
    managed_active = sorted(set(active) & set(ids))
    now = value["now_epoch"]
    reasons = []
    if managed_active != [value["pod_id"]] or value["gpu_count"] != 1:
        reasons.append("project-singleton-or-device-count")
    if revised:
        for k in ('terminate_after_request_epoch','watchdog_observed_epoch','watchdog_external_terminate_epoch'):
            integer(value[k],1,2**53-1,k)
        require_digest(value['watchdog_plan_sha256'])
        if (value['terminate_after_request_epoch'] != plan['provider_terminate_epoch'] or
                value['watchdog_plan_sha256'] != digest(plan) or
                value['watchdog_external_terminate_epoch'] != plan['external_terminate_epoch'] or
                not 0 <= now-value['watchdog_observed_epoch'] <= 30 or
                value['watchdog_state'] != 'ARMED'):
            reasons.append('missing-stale-or-changed-external-watchdog')
    elif (value["provider_guard_verified"] is not True or
            type(value["provider_terminate_epoch"]) is not int or
            value["provider_terminate_epoch"] != plan["provider_terminate_epoch"]):
        reasons.append("missing-or-changed-provider-deadline")
    if now < plan["input"]["now_epoch"] or not 0 <= now - value["observed_epoch"] <= 60:
        reasons.append("stale-or-inconsistent-provider-observation")
    if not 0 <= now - value["progress_epoch"] <= 300:
        reasons.append("stalled-or-future-progress")
    if not 0 <= now - value["last_checkpoint_epoch"] <= 1800:
        reasons.append("missing-recent-durable-checkpoint")
    hourly = money(value["hourly_usd"])
    if hourly <= 0 or hourly > money(plan["input"]["hourly_upper_usd"]):
        reasons.append("quote-upper-bound-exceeded")
    spent = money(value["actual_project_spend_usd"])
    outstanding = money(value["outstanding_usd"])
    reserved = money(value["reserved_remaining_usd"])
    if spent < money(plan["input"]["spent_usd"]):
        reasons.append("spend-regressed")
    # Keep mandatory remaining work reserved; reserves cannot silently shrink.
    if reserved < money(plan["input"]["reserved_remaining_usd"]):
        reasons.append("reservation-regressed")
    buffer = ceil_div((plan["input"]["checkpoint_grace_seconds"] + plan["input"].get('external_termination_grace_seconds',0) + plan["input"]["billing_slack_seconds"]) * max(hourly, money(plan["input"]["hourly_upper_usd"])), 3600)
    if spent + outstanding + reserved + buffer >= OPERATING_LIMIT:
        reasons.append("operating-budget-guard")
    if money(value["account_balance_usd"]) < outstanding + reserved + buffer + CAP - OPERATING_LIMIT:
        reasons.append("account-balance-insufficient-for-reserved-work")
    if now >= plan["request_checkpoint_epoch"]:reasons.append("checkpoint-deadline")
    hard_stop = revised and now >= plan['external_terminate_epoch']
    if hard_stop:reasons.append('external-termination-deadline')
    return {"schema": "ovl.supervisor-decision.v2" if revised else "ovl.supervisor-decision.v1", "plan_sha256": digest(plan),
            "observation_sha256": digest(value), "action": "TERMINATE" if hard_stop else "CHECKPOINT_AND_STOP" if reasons else "CONTINUE",
            "reasons": reasons, "managed_active_pod_ids": managed_active,
            "unrelated_active_pod_ids": sorted(set(active) - set(ids)),
            "provider_mutation": "NOT_RUN",
            **({'automatic_provider_termination':'UNVERIFIED'} if revised else {})}


class ControllerBusy(EvidenceError):
    """Another live owner holds the journal; a duplicate must not interfere."""


class Journal:
    """Owner-controlled append-only local journal; a hash chain is not public anchoring.

    The OS lock survives owner metadata changes and is released on process death.
    Never steal a lock based on a stale PID. Partial writes and gaps fail closed;
    preserve the damaged journal for explicit recovery instead of truncating it.
    """
    def __init__(self, directory: Path):
        self.directory = directory
        self._fd = None
        self.events = []

    @contextmanager
    def lease(self):
        if self._fd is not None:raise EvidenceError("controller lease already held")
        if self.directory.is_symlink():raise EvidenceError("controller directory is a symlink")
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        fd = os.open(self.directory / "controller.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:raise ControllerBusy("another controller holds the lease") from e
            self._fd = fd;self.events = self._read()
            yield self
        finally:
            self._fd = None;os.close(fd)

    def _read(self):
        events = [];previous = digest({"schema": "ovl.controller-journal.v1"})
        paths = sorted(self.directory.glob("event-*.json"))
        for index, path in enumerate(paths):
            if path.name != f"event-{index:08d}.json" or path.is_symlink():
                raise EvidenceError("controller journal gap or symlink")
            if path.stat().st_size > 1024 * 1024:raise EvidenceError("controller event oversized")
            event = parse_json(path.read_bytes(), canonical_required=True)
            fields(event, "schema sequence previous observed_epoch kind body", "controller event")
            integer(event["sequence"], 0, 99999999, "controller event sequence")
            integer(event["observed_epoch"], 1, 2**53 - 1, "controller event epoch")
            if event["schema"] != "ovl.controller-event.v1" or event["sequence"] != index or event["previous"] != previous:
                raise EvidenceError("controller journal ancestry mismatch")
            previous = digest(event);events.append(event)
        return events

    def append(self, kind, body):
        if self._fd is None:raise EvidenceError("controller event requires a live exclusive lease")
        if kind not in {"creation-intent", "creation-observed", "provider-observation", "decision", "checkpoint", "teardown", "failure"}:
            raise EvidenceError("unsupported controller event")
        event = {"schema": "ovl.controller-event.v1", "sequence": len(self.events),
                 "previous": digest(self.events[-1]) if self.events else digest({"schema": "ovl.controller-journal.v1"}),
                 "observed_epoch": int(time.time()), "kind": kind, "body": body}
        data = canonical(event)
        if len(data) > 1024 * 1024:raise EvidenceError("controller event oversized")
        path = self.directory / f"event-{len(self.events):08d}.json"
        # Publish only fully written/fsynced bytes. link() is atomic and refuses
        # to replace an existing event; a crash leaves either a complete event
        # or an ignored .pending-* draft, never an event-named torn write.
        fd, pending = tempfile.mkstemp(prefix=".pending-", dir=self.directory)
        with os.fdopen(fd, "wb") as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending, path, follow_symlinks=False)
        dir_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:os.fsync(dir_fd)
        finally:os.close(dir_fd)
        os.unlink(pending)  # Complete durable event remains; never the sole copy.
        self.events.append(event)
        return digest(event)
