"""Finite read recovery with actual local subprocesses and strict failure classes."""
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
import pod_observation_retry as m
from pod_transfer import EvidenceError,TransientTransportError,process_failure
from test_pod_transfer import setup


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
    (1,b'connection reset',False)])
def test_closed_ssh_failure_classification(code,diagnostic,transient):
    error=process_failure(code,diagnostic)
    assert isinstance(error,TransientTransportError)==transient
    assert diagnostic.decode() not in str(error)
