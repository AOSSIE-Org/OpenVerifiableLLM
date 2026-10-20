"""Tests for cross-platform host boot identity and monotonic boot clock."""
import re
import time
from pathlib import Path

import pytest

from ovl_pipeline.canonical import digest, host_boot_id, host_boottime_ms
from rental_safety import Lifetime, boot_clock


def test_host_boot_id_format():
    bid = host_boot_id()
    assert isinstance(bid, str)
    assert re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", bid), f"invalid boot id format: {bid}"


def test_host_boottime_ms_monotonic():
    t1 = host_boottime_ms()
    assert isinstance(t1, int)
    assert t1 > 0
    time.sleep(0.01)
    t2 = host_boottime_ms()
    assert t2 >= t1


def test_boot_clock_structure():
    clock = boot_clock()
    assert isinstance(clock, dict)
    assert set(clock.keys()) == {"boot_id", "boottime_ms"}
    assert isinstance(clock["boot_id"], str)
    assert isinstance(clock["boottime_ms"], int)
    assert clock["boottime_ms"] > 0
    assert re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", clock["boot_id"])


def test_host_boot_id_linux_mock(tmp_path, monkeypatch):
    synthetic_uuid = "12345678-abcd-ef01-2345-6789abcdef01"
    fake_proc = tmp_path / "proc_boot_id"
    fake_proc.write_text(synthetic_uuid + "\n")

    orig_is_file = Path.is_file
    orig_read_text = Path.read_text

    def mock_is_file(self):
        if str(self) == "/proc/sys/kernel/random/boot_id":
            return True
        return orig_is_file(self)

    def mock_read_text(self, *args, **kwargs):
        if str(self) == "/proc/sys/kernel/random/boot_id":
            return synthetic_uuid
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", mock_is_file)
    monkeypatch.setattr(Path, "read_text", mock_read_text)

    assert host_boot_id() == synthetic_uuid


def test_lifetime_initialization_with_real_boot_clock():
    class MockJournal:
        def __init__(self):
            self.events = []
        def append(self, kind, body):
            self.events.append({'kind': kind, 'body': body})

    now = int(time.time())
    plan = {
        'input': {'now_epoch': now},
        'external_terminate_epoch': now + 300,
    }
    journal = MockJournal()
    lt = Lifetime(journal, plan, initialize=True)
    assert lt.anchor is not None
    assert lt.anchor['action'] == 'LIFETIME_CLOCK'
    assert lt.anchor['plan_sha256'] == digest(plan)
    assert re.fullmatch(r'[a-zA-Z0-9-]{1,96}', lt.anchor['boot_id'])
    assert lt.remaining() > 0
