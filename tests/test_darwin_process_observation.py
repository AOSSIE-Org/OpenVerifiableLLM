"""Tests for Darwin process observation, zombie telemetry, and group lifecycle."""
import os
import signal
import subprocess
import sys
import time
import pytest

from ovl_pipeline.canonical import host_process_stat
sys.path.insert(0, str(os.path.abspath("scripts")))
import pod_job_worker as worker


def test_host_process_stat_current_process():
    state, ticks, pgid = host_process_stat()
    assert state in ("R", "S")
    assert ticks.isdecimal()
    assert int(ticks) > 0
    assert pgid == os.getpgrp()

    # Explicit PID
    state_pid, ticks_pid, pgid_pid = host_process_stat(os.getpid())
    assert (state_pid, ticks_pid, pgid_pid) == (state, ticks, pgid)


def test_host_process_stat_nonexistent_pid_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        host_process_stat(9999999)


def test_host_process_stat_preserves_ticks_and_pgid_on_zombie():
    pid = os.fork()
    if pid == 0:
        time.sleep(0.05)
        os._exit(0)

    try:
        # Check while running
        run_state, run_ticks, run_pgid = host_process_stat(pid)
        assert run_state in ("R", "S")
        assert run_pgid == os.getpgrp()

        # Wait for child to exit and become a zombie
        time.sleep(0.1)
        zomb_state, zomb_ticks, zomb_pgid = host_process_stat(pid)
        assert zomb_state == "Z"
        assert zomb_ticks == run_ticks
        assert zomb_pgid == run_pgid
    finally:
        os.waitpid(pid, 0)


def test_worker_process_identity():
    ident = worker.process_identity(os.getpid())
    assert ident["pid"] == os.getpid()
    assert isinstance(ident["start_ticks"], int)
    assert ident["start_ticks"] > 0
    assert ident["process_group"] == os.getpgrp()

    with pytest.raises(FileNotFoundError):
        worker.process_identity(9999999)


def test_worker_alive_and_exit_ready_lifecycle():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        start_new_session=True,
    )
    ident = worker.process_identity(child.pid)
    try:
        assert worker.alive(ident) is True
        assert worker.exit_ready(child.pid) is False

        # Wait for child to finish execution
        time.sleep(0.3)
        assert worker.exit_ready(child.pid) is True
        assert worker.alive(ident) is False

        # Child is still unreaped because exit_ready used WNOWAIT
        reaped_pid, status = os.waitpid(child.pid, os.WNOHANG)
        assert reaped_pid == child.pid
        assert os.waitstatus_to_exitcode(status) == 0

        # After reaping, alive returns False
        assert worker.alive(ident) is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)


def test_worker_group_alive():
    assert worker.group_alive(os.getpgrp()) is True
    assert worker.group_alive(9999999) is False


def test_worker_signal_owned_zombie_and_identity_mismatch():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        start_new_session=True,
    )
    ident = worker.process_identity(child.pid)
    try:
        # Wait for child to exit
        time.sleep(0.1)
        assert worker.exit_ready(child.pid) is True

        # Signaling the zombie leader/group succeeds without raising PermissionError or Refusal
        assert worker.signal_owned(ident, signal.SIGKILL) is True

        # Modified identity refused
        corrupted = dict(ident, start_ticks=ident["start_ticks"] + 1)
        with pytest.raises(worker.Refusal, match="process identity changed"):
            worker.signal_owned(corrupted, signal.SIGKILL)
    finally:
        child.wait(timeout=2)
