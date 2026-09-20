"""Finite read recovery with actual local subprocesses and strict failure classes."""
from pathlib import Path
import subprocess
import shutil
import socket
import sys
import time
from types import SimpleNamespace
import pytest
import pod_observation_retry as m
from pod_transfer import EvidenceError,TransientTransportError,process_failure
from test_pod_transfer import setup


def test_real_ssh_refusal_reaches_bounded_read_recovery(tmp_path):
    """Exercise actual SSH flags: a subprocess double cannot catch -q hiding errors."""
    if shutil.which('ssh') is None:pytest.skip('OpenSSH client required')
    t,remote,calls,processes=setup(tmp_path)
    (remote/'status.json').write_text('{"status":"running"}')
    original=t.popen;attempts=[]
    # Holding a bound, non-listening loopback socket prevents another listener
    # taking this port; no remote service, real credential or host key is used.
    with socket.socket() as closed:
        closed.bind(('127.0.0.1',0));t.profile['port']=closed.getsockname()[1]
        def flaky(command,**kw):
            attempts.append(command)
            if len(attempts)==1:
                p=subprocess.Popen(command,**kw);processes.append(p);return p
            return original(command,**kw)
        t.popen=flaky;health=Health();deadline=health.plan['external_terminate_epoch']
        value=m.read('metadata',t,{'status.json':tmp_path/'observed.json'},health,tmp_path/'health',tmp_path,sleep=lambda n:None)
    assert value=={'status.json':{'status':'running'}} and len(attempts)==2
    assert health.plan['external_terminate_epoch']==deadline
    failure=(tmp_path/'metadata-transport-failure-0.json').read_text()
    assert 'transient-transport' in failure and '127.0.0.1' not in failure
    assert not (tmp_path/'metadata-transport-failure-1.json').exists()
    assert all(p.poll() is not None for p in processes)


class Health:
    def __init__(self,deadline=None):
        self.plan={'external_terminate_epoch':deadline or int(time.time())+60};self.writes=0
    def now(self):return int(time.time())
    def write(self,path):self.writes+=1


def test_actual_metadata_read_recovers_once_and_preserves_failure(tmp_path):
    t,remote,calls,processes=setup(tmp_path);(remote/'status.json').write_text('{"status":"running"}')
    original=t.popen;attempts=[]
    def flaky(command,**kw):
        attempts.append(command)
        if len(attempts)==1:
            p=subprocess.Popen([sys.executable,'-c','import sys;sys.stderr.write("Connection reset by peer");sys.exit(255)'],**kw)
            processes.append(p);return p
        return original(command,**kw)
    t.popen=flaky;health=Health()
    value=m.read('metadata',t,{'status.json':tmp_path/'observed.json'},health,tmp_path/'health',tmp_path,sleep=lambda n:None)
    assert value=={'status.json':{'status':'running'}} and len(attempts)==2
    assert (tmp_path/'metadata-transport-failure-0.json').is_file()
    assert not (tmp_path/'metadata-transport-failure-1.json').exists()
    assert all(p.poll() is not None for p in processes)
    assert health.writes>=3


@pytest.mark.parametrize('error',[EvidenceError('hash mismatch'),EvidenceError('host-key selection mismatch')])
def test_integrity_failures_never_retry(tmp_path,monkeypatch,error):
    calls=[]
    def bad(*args):calls.append(args);raise error
    monkeypatch.setattr(m,'observe_many',bad)
    with pytest.raises(EvidenceError):m.read('metadata',None,{},Health(),tmp_path/'health',tmp_path)
    assert len(calls)==1 and not list(tmp_path.glob('*failure*'))


def test_second_transport_failure_stops_and_deadline_never_moves(tmp_path,monkeypatch):
    health=Health();fixed=health.plan['external_terminate_epoch'];calls=[]
    def bad(*args):calls.append(args[-1]);raise TransientTransportError('synthetic')
    monkeypatch.setattr(m,'job_supervision',bad)
    with pytest.raises(TransientTransportError):m.read('supervision',None,('a','b'),health,tmp_path/'h',tmp_path,sleep=lambda n:None)
    assert len(calls)==2 and all(d<=fixed for d in calls)
    assert len(list(tmp_path.glob('*failure*')))==2
    assert health.plan['external_terminate_epoch']==fixed


@pytest.mark.parametrize('external,expected',[(1100,[1020,1042]),(1030,[1020,1030])])
def test_advancing_clock_caps_attempts_and_backoff(tmp_path,monkeypatch,external,expected):
    class ClockHealth(Health):
        clock=1000
        def now(self):return self.clock
    health=ClockHealth(external);calls=[];sleeps=[]
    def observe(*args):
        calls.append(args[-1])
        if len(calls)==1:
            health.clock+=19;raise TransientTransportError('synthetic')
        return {'synthetic':'second complete observation'}
    def sleep(seconds):sleeps.append(seconds);health.clock+=seconds
    monkeypatch.setattr(m,'job_supervision',observe)
    assert m.read('supervision',None,('a','b'),health,tmp_path/'h',tmp_path,sleep=sleep)=={'synthetic':'second complete observation'}
    assert calls==expected and sleeps==[3] and health.plan['external_terminate_epoch']==external


def test_backoff_oversleep_cannot_start_read_after_fixed_deadline(tmp_path,monkeypatch):
    class ClockHealth(Health):
        clock=1000
        def now(self):return self.clock
    health=ClockHealth(1004);calls=[]
    def observe(*args):calls.append(args[-1]);raise TransientTransportError('synthetic')
    def sleep(seconds):assert seconds==3;health.clock+=5
    monkeypatch.setattr(m,'job_supervision',observe)
    with pytest.raises(EvidenceError,match='original observation deadline expired'):
        m.read('supervision',None,('a','b'),health,tmp_path/'h',tmp_path,sleep=sleep)
    assert calls==[1004] and health.plan['external_terminate_epoch']==1004


def test_no_retry_at_original_deadline_or_mutation(tmp_path,monkeypatch):
    calls=[]
    def bad(*args):calls.append(args);raise TransientTransportError('synthetic')
    monkeypatch.setattr(m,'job_supervision',bad)
    for kind in ('abandon','put','launch'):
        with pytest.raises(EvidenceError):m.read(kind,None,(),Health(),tmp_path/'h',tmp_path)
    assert calls==[]
    with pytest.raises(TransientTransportError):
        m.read('supervision',None,('a','b'),Health(int(time.time())+2),tmp_path/'h',tmp_path,sleep=lambda n:pytest.fail('late retry'))
    assert len(calls)==1


@pytest.mark.parametrize('code,diagnostic,transient',[
    (255,b'Connection timed out',True),(255,b'Connection reset by peer',True),
    (255,b'Connection closed; HOST KEY VERIFICATION FAILED',False),
    (255,b'Permission denied; connection closed',False),(255,b'unknown failure',False),
    (255,b'Authentication failed; connection refused',False),
    (255,b'Host identification has changed; connection refused',False),
    (255,b'Load key "synthetic-key": invalid format\nConnection closed',False),
    (255,b'sign_and_send_pubkey: signing failed\nConnection reset by peer',False),
    (255,b'unknown failure\nconnection reset',False),
    (255,b'connection refused; unknown failure',False),
    (255,b'ssh: connect to host 127.0.0.1 port 2222: Connection refused\r\n',True),
    (255,b'kex_exchange_identification: read: Connection reset by peer\r\n',True),
    (255,b'',False),
    (1,b'connection reset',False)])
def test_closed_ssh_failure_classification(code,diagnostic,transient):
    error=process_failure(code,diagnostic)
    assert isinstance(error,TransientTransportError)==transient
    if diagnostic:assert diagnostic.decode() not in str(error)


@pytest.mark.parametrize('diagnostic',['Permission denied','Load key: invalid format','unknown failure'])
def test_strict_error_before_timeout_never_retries(tmp_path,diagnostic):
    t,remote,calls,processes=setup(tmp_path);attempts=[]
    def hang(command,**kw):
        attempts.append(command)
        p=subprocess.Popen([sys.executable,'-c','import sys,time;sys.stderr.write(sys.argv[1]);sys.stderr.flush();time.sleep(15)',diagnostic],**kw)
        processes.append(p);return p
    t.popen=hang
    with pytest.raises(EvidenceError) as caught:
        m.read('metadata',t,{'status.json':tmp_path/'observed.json'},Health(int(time.time())+2),tmp_path/'h',tmp_path,sleep=lambda n:pytest.fail('strict failure retried'))
    assert not isinstance(caught.value,TransientTransportError) and len(attempts)==1
    assert diagnostic not in str(caught.value) and not list(tmp_path.glob('*transport-failure*'))
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)


@pytest.mark.parametrize('prefix',['','Connection reset by peer\n'])
def test_complete_split_denial_fails_promptly_under_long_deadline(tmp_path,prefix):
    t,remote,calls,processes=setup(tmp_path);attempts=[]
    def denial(command,**kw):
        attempts.append(command)
        script='import os,time,sys;os.write(2,sys.argv[1].encode()+b"Permiss");time.sleep(.05);os.write(2,b"ion denied\\n");time.sleep(15)'
        p=subprocess.Popen([sys.executable,'-c',script,prefix],**kw);processes.append(p);return p
    t.popen=denial;started=time.monotonic()
    with pytest.raises(EvidenceError) as error:
        m.read('metadata',t,{'status.json':tmp_path/'observed.json'},Health(int(time.time())+30),tmp_path/'h',tmp_path,sleep=lambda n:pytest.fail('denial retried'))
    assert time.monotonic()-started<2 and len(attempts)==1
    assert not isinstance(error.value,TransientTransportError)
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)
