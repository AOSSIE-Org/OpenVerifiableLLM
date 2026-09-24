"""Scheduled pipe doubles exercise the real stream timer without wall-clock waits."""
import io
import shlex
import subprocess
from types import SimpleNamespace

import pytest

import pod_transfer as m
from ovl_pipeline.canonical import EvidenceError, sha256
from test_pod_transfer import setup


def scheduled(tmp_path, monkeypatch, schedule):
    transport, remote, calls, processes = setup(tmp_path)
    clock = [1000.0]
    channels = {}
    current = []
    real_read, real_blocking = m.os.read, m.os.set_blocking

    class Channel:
        def __init__(self):
            self.fd = 100000 + len(channels)
            self.closed = False
            self.ready = []
            channels[self.fd] = self

        def fileno(self): return self.fd
        def close(self): self.closed = True

    class Process:
        def __init__(self, command):
            self.stdin = None
            self.stdout, self.stderr = Channel(), Channel()
            self.pid = 900000 + len(processes)
            self.returncode = None
            events, self.exits = schedule(shlex.split(command[-1])[1:])
            self.events = [(clock[0] + delay, getattr(self, channel), data)
                           for delay, channel, data in events]

        def poll(self): return self.returncode

        def wait(self, timeout):
            if self.returncode is not None: return self.returncode
            clock[0] += timeout
            raise subprocess.TimeoutExpired('synthetic-owned-process', timeout)

    class Selector:
        def __init__(self): self.mapping = {}
        def register(self, channel, flags, data):
            self.mapping[channel.fd] = SimpleNamespace(fileobj=channel, data=data)
        def unregister(self, channel): del self.mapping[channel.fd]
        def get_map(self): return self.mapping
        def close(self): self.mapping.clear()

        def select(self, timeout):
            process = current[0]
            if not process.events or process.events[0][0] > clock[0] + timeout:
                clock[0] += timeout
                return []
            when, channel, data = process.events.pop(0)
            clock[0] = max(clock[0], when)
            channel.ready.append(data)
            if not process.events and process.exits: process.returncode = 0
            return [(self.mapping[channel.fd], m.selectors.EVENT_READ)]

    def popen(command, **kwargs):
        process = Process(command)
        processes.append(process)
        current[:] = [process]
        calls.append(command)
        return process

    def read(fd, amount):
        if fd not in channels: return real_read(fd, amount)
        channel = channels[fd]
        if not channel.ready: raise BlockingIOError()
        data = channel.ready.pop(0)
        assert len(data) <= amount
        return data

    def kill(pid, signal):
        process = next(p for p in processes if p.pid == pid)
        process.returncode = -signal

    monkeypatch.setattr(m.os, 'read', read)
    monkeypatch.setattr(m.os, 'set_blocking', lambda fd, flag: None if fd in channels else real_blocking(fd, flag))
    monkeypatch.setattr(m.os, 'killpg', kill)
    monkeypatch.setattr(m.selectors, 'DefaultSelector', Selector)
    transport.popen = popen
    transport.wall = transport.monotonic = lambda: clock[0]
    return transport, clock, processes


def test_productive_ranges_exceed_ninety_seconds_and_finish_original_deadline(tmp_path, monkeypatch):
    data = bytes(range(32))
    monkeypatch.setattr(m, 'RANGE_BYTES', 16)

    def schedule(argv):
        offset, length = map(int, argv[-2:])
        events = [(8 * (i + 1), 'stdout', data[offset+i:offset+i+1]) for i in range(length)]
        return events + [(8 * length, 'stdout', b''), (8 * length, 'stderr', b'')], True

    transport, clock, processes = scheduled(tmp_path, monkeypatch, schedule)
    result = transport.get('state', tmp_path/'download', {'path':'state','bytes':32,'sha256':sha256(data)}, 1400)
    assert clock[0] == 1256 and (tmp_path/'download').read_bytes() == data
    assert len(processes) == 2 and all(p.returncode == 0 for p in processes)
    assert result['range_policy']['payload_idle_seconds'] == 90
    assert result['range_policy']['original_deadline_epoch'] == 1400


@pytest.mark.parametrize('mode,elapsed,received', [('silence',90,0), ('prefix',91,1), ('exit-hang',91,1), ('stderr',90,0), ('trickle',200,2)])
def test_inactivity_exit_and_parent_limits_reap_owned_process(tmp_path, monkeypatch, mode, elapsed, received):
    def schedule(argv):
        if mode == 'silence': return [], False
        if mode == 'prefix': return [(1,'stdout',b'x')], False
        if mode == 'exit-hang': return [(1,'stdout',b'x'),(2,'stdout',b''),(2,'stderr',b'')], False
        if mode == 'stderr': return [(x,'stderr',b'\n') for x in (30,60,89,119)], False
        return [(x,'stdout',b'x') for x in (80,160,240,320)], False

    transport, clock, processes = scheduled(tmp_path, monkeypatch, schedule)
    # Unknown diagnostics (including whitespace-only output) stay strict.
    category = EvidenceError if mode == 'stderr' else m.TransientTransportError
    with pytest.raises(category) as error:
        transport.stream(['/bin/cat','fixture'], io.BytesIO(), 10, 1200, payload_idle_seconds=90)
    assert type(error.value) is category
    assert clock[0] == 1000 + elapsed
    assert error.value.transfer_counts['bytes_received'] == received
    assert all(p.returncode is not None and p.stdout.closed and p.stderr.closed for p in processes)


def test_inactivity_does_not_delay_ambiguous_denial_failure(tmp_path, monkeypatch):
    transport, clock, processes = scheduled(tmp_path, monkeypatch,
        lambda argv: ([(1,'stderr',b'Permission denied\n')], False))
    with pytest.raises(EvidenceError, match='ambiguous SSH or remote denial') as error:
        transport.stream(['/bin/cat','fixture'], io.BytesIO(), 10, 1400, payload_idle_seconds=90)
    assert type(error.value) is EvidenceError and clock[0] == 1001
    assert len(processes) == 1 and processes[0].returncode is not None


@pytest.mark.parametrize('latency',[0,500])
def test_callback_latency_cannot_hide_parent_deadline_crossing(tmp_path,monkeypatch,latency):
    transport,clock,processes=scheduled(tmp_path,monkeypatch,
        lambda argv: ([(1,'stdout',b'x'),(1,'stdout',b''),(1,'stderr',b'')],True))
    seen=[]
    def progress(counts):
        seen.append(dict(counts));clock[0]+=latency
    if latency:
        with pytest.raises(EvidenceError,match='outside transfer deadline'):
            transport.stream(['/bin/cat','fixture'],io.BytesIO(),1,1200,progress=progress,payload_idle_seconds=90)
    else:
        assert transport.stream(['/bin/cat','fixture'],io.BytesIO(),1,1200,progress=progress,payload_idle_seconds=90)['bytes_received']==1
    assert seen==[{'bytes_sent':0,'bytes_received':1}]
    assert clock[0]==1001+latency and all(p.returncode is not None for p in processes)


@pytest.mark.parametrize('allowance',[0,3601,True,1.5])
def test_invalid_idle_allowance_cannot_start_transport(tmp_path,monkeypatch,allowance):
    transport,clock,processes=scheduled(tmp_path,monkeypatch,lambda argv:([],False))
    with pytest.raises(EvidenceError):
        transport.stream(['/bin/cat','fixture'],io.BytesIO(),1,1200,payload_idle_seconds=allowance)
    assert not processes


def test_payload_idle_policy_cannot_reclassify_upload(tmp_path,monkeypatch):
    transport,clock,processes=scheduled(tmp_path,monkeypatch,lambda argv:([],False))
    with pytest.raises(EvidenceError,match='read-only'):
        transport.stream(['/bin/cat','fixture'],io.BytesIO(),1,1200,source=io.BytesIO(b'x'),source_bytes=1,payload_idle_seconds=90)
    assert not processes
