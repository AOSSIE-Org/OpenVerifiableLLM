"""Durable single-host lifecycle primitives, independent of scientific kernels.

The journal is private operational state, never a publication inventory. External
effects must be reconciled by their original identity after uncertain delivery.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import time
from typing import Callable, Protocol

from .canonical import EvidenceError, digest, read_json, write_json


class RetryableRead(OSError):
    """Adapter has positively classified a transient read/transport failure."""


class Pending(RuntimeError):
    """An original operation needs more observation; it must not be resubmitted."""


@dataclass(frozen=True)
class Observation:
    status: str  # absent, running, complete, unknown
    result: dict | None = None
    # True only when the adapter proves the ORIGINAL request was not accepted.
    # A momentarily empty eventual-consistency listing is insufficient.
    absence_proven: bool = False


class Effect(Protocol):
    def observe(self, operation: str, request: dict) -> Observation: ...
    def submit(self, operation: str, request: dict) -> None: ...
    def validate(self, operation: str, request: dict, result: dict) -> None: ...


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path):
    """Persist every newly created directory entry before publishing effects."""
    path = Path(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise EvidenceError("invalid directory ancestry")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise EvidenceError('concurrent directory creation has invalid type')
        sync_directory(directory.parent)


@contextmanager
def exclusive(directory: Path):
    """Hold one owner across local computation and external reconciliation."""
    directory = Path(directory)
    durable_mkdir(directory)
    if directory.is_symlink():
        raise EvidenceError("journal directory must not be a symlink")
    fd = os.open(directory / "owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Pending("another lifecycle owner holds the lock") from exc
        yield fd
    finally:
        os.close(fd)


class Journal:
    """One atomic state file. Use only while holding exclusive(directory)."""

    def __init__(self, directory: Path, identity: dict):
        self.directory = Path(directory)
        self.path = self.directory / "journal.json"
        self.state = {"schema": "ovl.lifecycle-journal.v1", "identity": identity, "operations": []}
        if self.path.exists():
            old = read_json(self.path)
            if set(old) != set(self.state) or old["schema"] != self.state["schema"] or old["identity"] != identity:
                raise EvidenceError("lifecycle identity changed")
            if type(old["operations"]) is not list:
                raise EvidenceError("invalid operation journal")
            names = set()
            for item in old["operations"]:
                if (set(item) != {"name", "id", "request", "status", "result"}
                        or item["status"] not in ("intent", "sent", "complete")
                        or item["name"] in names
                        or item["id"] != digest({"identity": identity, "name": item["name"], "request": item["request"]})):
                    raise EvidenceError("invalid or conflicting operation")
                names.add(item["name"])
            self.state = old
        else:
            if any(p.name != "owner.lock" for p in self.directory.iterdir()):
                raise EvidenceError("missing journal beside prior operational evidence")
            self.save()

    def save(self):
        write_json(self.path, self.state)

    def operation(self, name, request):
        wanted = digest({"identity": self.state["identity"], "name": name, "request": request})
        for item in self.state["operations"]:
            if item["name"] == name:
                if item["id"] != wanted:
                    raise EvidenceError("operation input changed; a new run is required")
                return item
        if any(x["status"] != "complete" for x in self.state["operations"]):
            raise Pending("previous operation has not completed")
        item = {"name": name, "id": wanted, "request": request, "status": "intent", "result": None}
        self.state["operations"].append(item)
        self.save()
        return item

    def local(self, name: str, request: dict, compute: Callable, validate: Callable, *, event=lambda _: None):
        """Local deterministic stage; compute owns validated recovery of partials.

        No remote mutations are permitted in compute. A completed receipt is
        checked again on every adoption. Failures preserve intent and all bytes.
        """
        item = self.operation(name, request)
        if item["status"] == "complete":
            validate(item["result"])
            return item["result"]
        item["status"] = "sent"
        self.save()
        event(name + ":intent")
        result = compute(item["id"])
        validate(result)
        event(name + ":result")
        item.update(status="complete", result=result)
        self.save()
        event(name + ":committed")
        return result

    def effect(self, name: str, request: dict, adapter: Effect, *, deadline: float,
               clock=time.time, sleep=time.sleep, read_window=120, event=lambda _: None):
        """At most one uncertain submission; retries apply only to typed reads."""
        item = self.operation(name, request)
        if item["status"] == "complete" and hasattr(adapter, "adopt"):
            # Immutable results must be checked by their recorded revision,
            # independently of a provider's mutable current HEAD/listing.
            adapter.adopt(item["id"], request, item["result"])
            return item["result"]
        last_success = clock()
        delay = 1
        submitted_here = False
        while clock() < deadline:
            try:
                observation = adapter.observe(item["id"], request)
                last_success = clock()
                delay = 1
            except RetryableRead:
                end = min(deadline, last_success + read_window)
                if clock() + delay >= end:
                    raise Pending("read recovery window exhausted; original operation retained")
                sleep(delay)
                delay = min(delay * 2, 10)
                continue
            if observation.status == "complete":
                if type(observation.result) is not dict:
                    raise EvidenceError("missing effect result")
                adapter.validate(item["id"], request, observation.result)
                if item["status"] == "complete" and item["result"] != observation.result:
                    raise EvidenceError("completed effect identity changed")
                item.update(status="complete", result=observation.result)
                self.save()
                return observation.result
            if item["status"] == "complete":
                raise EvidenceError("previously completed effect is no longer verifiable")
            if observation.status == "absent":
                if submitted_here or (item["status"] == "sent" and not observation.absence_proven):
                    raise Pending("uncertain mutation; reconcile original operation before retry")
                item["status"] = "sent"
                self.save()  # BEFORE any possible external side effect
                event(name + ":intent")
                if clock() >= deadline:
                    raise Pending("effect deadline passed before submission; reconcile retained intent")
                submitted_here = True
                adapter.submit(item["id"], request)
                event(name + ":submitted")
                continue
            if observation.status == "unknown":
                raise Pending("unknown operation identity; no resubmission")
            if observation.status != "running":
                raise EvidenceError("invalid effect observation")
            if item["status"] != "sent":
                item["status"] = "sent"
                self.save()
            if clock() + delay >= deadline:
                break
            sleep(delay)
        raise Pending("operation deadline reached; independent shutdown guard must reconcile")


def reconcile_billing(resources: list[dict], previous: dict, rows: list[dict]):
    """Cumulative snapshots replace prior amounts; each resource retains its cap.

    Inputs are decimal strings, never binary floats. This does not assert final
    provider settlement or release an unposted liability without closure evidence.
    """
    ceilings = {}
    for item in resources:
        ident = item["id"]
        upper = Decimal(item["ceiling"])
        if ident in ceilings or not upper.is_finite() or upper < 0:
            raise EvidenceError("invalid or duplicate resource ceiling")
        ceilings[ident] = upper
    amounts = {k: Decimal(v) for k, v in previous.items()}
    if set(amounts) - set(ceilings):
        raise EvidenceError("unattributed billing baseline")
    seen = set()
    for row in rows:
        ident, value = row["id"], Decimal(row["cumulative"])
        if ident not in ceilings or ident in seen or not value.is_finite() or value < amounts.get(ident, 0):
            raise EvidenceError("unattributed, duplicate or decreasing bill")
        seen.add(ident)
        amounts[ident] = value
    for ident, value in amounts.items():
        if not value.is_finite() or not 0 <= value <= ceilings[ident]:
            raise EvidenceError("billing exceeds reserved resource ceiling")
    posted = sum(amounts.values(), Decimal(0))
    upper = sum(ceilings.values(), Decimal(0))
    return {"posted": str(posted), "reserved": str(upper-posted), "upper": str(upper),
            "by_resource": {k: str(v) for k, v in sorted(amounts.items())}, "settled": False}
