"""Synthetic timeout diagnostics; no provider or account inputs."""
import subprocess
import io
import sys
import time

import pytest

import pod_transfer as m
from ovl_pipeline.canonical import EvidenceError, read_json, sha256
from test_pod_transfer import setup


TIMEOUT = b'Timeout, server 127.0.0.1 not responding.\r\n'


@pytest.mark.parametrize('code,message,retry', [
    (255, TIMEOUT, True),
    (1, TIMEOUT, False),
    (255, b'', False),
    (255, TIMEOUT + b'Permission denied\n', False),
    (255, b'Host key verification failed\n' + TIMEOUT, False),
    (255, TIMEOUT + b'unknown failure\n', False),
    (255, TIMEOUT.rstrip() + b' ignored suffix\n', False),
    (255, b'Timeout, server arbitrary-host not responding.\n', False),
])
def test_only_closed_keepalive_diagnostic_is_retryable(code, message, retry):
    error = m.process_failure(code, message)
    assert (type(error) is m.TransientTransportError) == retry
    assert '127.0.0.1' not in str(error)


def test_selected_command_keeps_info_without_verbose_auth_output(tmp_path):
    transport, _, _, _ = setup(tmp_path)
    command = transport.command(['/bin/true'])
    assert 'LogLevel=INFO' in command
    assert 'LogLevel=ERROR' not in command and '-q' not in command
    assert 'ServerAliveInterval=10' in command
    assert 'ServerAliveCountMax=2' in command


@pytest.mark.parametrize('fault', ['recover', 'exhaust', 'empty', 'denial', 'conflict'])
@pytest.mark.parametrize('timeout_message',[TIMEOUT,b'Connection to 127.0.0.1 port 2222 timed out\r\n'])
def test_partial_range_keepalive_failure_preserves_checks(tmp_path, monkeypatch, fault,timeout_message):
    transport, remote, calls, processes = setup(tmp_path)
    monkeypatch.setattr(m, 'RANGE_BYTES', 16)
    data = b'x' * 32
    (remote/'state').write_bytes(data)
    expected = {'path':'state', 'bytes':32, 'sha256':sha256(data)}
    original = transport.popen
    failures = []
    deadlines = []
    stream = transport.stream
    def observe(*args, **kwargs):
        deadlines.append(args[3])
        return stream(*args, **kwargs)
    transport.stream = observe
    def fail(command, **kwargs):
        if not failures or fault == 'exhaust':
            failures.append(True)
            message = b'' if fault == 'empty' else timeout_message
            if fault == 'denial': message += b'Permission denied\n'
            prefix = b'z' * 8 if fault == 'conflict' else data[:8]
            child = subprocess.Popen([sys.executable, '-c',
                'import os,sys;os.write(1,bytes.fromhex(sys.argv[1]));os.write(2,bytes.fromhex(sys.argv[2]));sys.exit(255)',
                prefix.hex(), message.hex()], **kwargs)
            processes.append(child)
            return child
        return original(command, **kwargs)
    transport.popen = fail
    deadline = int(time.time()) + 30
    if fault == 'recover':
        result = transport.get('state', tmp_path/'download', expected, deadline)
        assert (tmp_path/'download').read_bytes() == data
        assert result['bytes_received'] == 32 and result['transferred_payload_bytes'] == 40
        assert len(processes) == 3
    else:
        category = m.RangeRecoveryExhausted if fault == 'exhaust' else EvidenceError
        with pytest.raises(category):
            transport.get('state', tmp_path/'download', expected, deadline)
        assert not (tmp_path/'download').exists()
        assert len(processes) == (3 if fault == 'exhaust' else 2 if fault == 'conflict' else 1)
    assert all(d <= deadline for d in deadlines)
    assert (tmp_path/'download.partial.ranges/attempt-00000.partial').read_bytes() == (b'z'*8 if fault == 'conflict' else data[:8])
    receipt = read_json(tmp_path/'download.partial.ranges/attempt-00000.json')
    assert receipt['bytes_received'] == receipt['saved_bytes'] == 8
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)


@pytest.mark.parametrize('message', [b'Connection reset', TIMEOUT.rstrip()])
def test_unfinished_running_diagnostic_never_grants_retry(tmp_path, monkeypatch, message):
    from test_payload_inactivity import scheduled
    transport, clock, processes = scheduled(tmp_path, monkeypatch,
        lambda argv: ([(1,'stderr',message)], False))
    with pytest.raises(EvidenceError) as caught:
        transport.stream(['/bin/true'], io.BytesIO(), 1, 1200, payload_idle_seconds=90)
    assert type(caught.value) is EvidenceError and clock[0] == 1090
    assert all(p.poll() is not None for p in processes)


@pytest.mark.parametrize('message', [b'Permission denied', b'Load key synthetic: invalid format', b'sign_and_send_pubkey: signing failed'])
@pytest.mark.parametrize('newline', [b'', b'\n'])
def test_eof_denial_cannot_be_hidden_by_success_exit_and_matching_payload(tmp_path, message, newline):
    transport, _, _, processes = setup(tmp_path)
    def child(command, **kwargs):
        process = subprocess.Popen([sys.executable, '-c',
            'import os,sys;os.write(1,b"x");os.write(2,bytes.fromhex(sys.argv[1]))',
            (message+newline).hex()], **kwargs)
        processes.append(process)
        return process
    transport.popen = child
    expected='ambiguous SSH or remote denial' if message==b'Permission denied' else 'authentication or identity'
    with pytest.raises(EvidenceError, match=expected):
        transport.get('state', tmp_path/'download', {'path':'state','bytes':1,'sha256':sha256(b'x')}, int(time.time())+30)
    assert not (tmp_path/'download').exists()
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)


@pytest.mark.parametrize('message,retry', [
    (b'Connection to 127.0.0.1 port 2222 timed out\r\n', True),
    (b'Connection to 127.0.0.1 port 2222 timed out\nPermission denied\n', False),
    (b'Host key verification failed\nConnection to 127.0.0.1 port 2222 timed out\n', False),
    (b'Connection to arbitrary-host port 2222 timed out\n', False),
    (b'Connection to 127.0.0.1 port 2222 timed out extra\n', False),
    (b'Connection to 127.0.0.1 port 2222 timed out\nunknown error\n', False),
])
def test_numeric_endpoint_timeout_retains_closed_diagnostic_policy(message, retry):
    assert (type(m.process_failure(255, message)) is m.TransientTransportError) == retry
    assert type(m.process_failure(1, message)) is EvidenceError
