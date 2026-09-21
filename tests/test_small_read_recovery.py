"""Bounded zero-payload recovery for selected immutable metadata, without credit."""
import json
import subprocess
import sys
import time

import pytest

from test_pod_transfer import setup
import pod_transfer as m
from ovl_pipeline.canonical import EvidenceError, sha256


def fixture(tmp_path,monkeypatch):
    t,remote,calls,processes=setup(tmp_path)
    expected={'path':'updates.jsonl','bytes':3,'sha256':sha256(b'abc')}
    (remote/expected['path']).write_bytes(b'abc')
    clock=[1000.0];t.wall=lambda:clock[0];t.monotonic=lambda:clock[0]
    sleeps=[]
    def sleep(n):sleeps.append(n);clock[0]+=n
    monkeypatch.setattr(m.time,'sleep',sleep)
    return t,remote,expected,clock,sleeps


def transient():
    e=m.TransientTransportError('synthetic connection timeout')
    e.transfer_counts={'bytes_sent':0,'bytes_received':0}
    return e


@pytest.mark.parametrize('failure_count',[1,2])
def test_recovers_exact_read_without_renewal_or_payload_credit(tmp_path,monkeypatch,failure_count):
    t,remote,expected,clock,sleeps=fixture(tmp_path,monkeypatch);attempts=[];events=[]
    def stream(argv,dest,maximum,deadline,**kw):
        attempts.append((argv,maximum,deadline));clock[0]+=15
        if len(attempts)<=failure_count:raise transient()
        dest.write(b'abc');kw['progress']({'bytes_sent':0,'bytes_received':3})
        return {'bytes_sent':0,'bytes_received':3,'process_exit_code':0}
    t.stream=stream
    got=t.get(expected['path'],tmp_path/'download',expected,1100,progress=events.append)
    assert (tmp_path/'download').read_bytes()==b'abc' and events==[{'bytes_sent':0,'bytes_received':3}]
    assert len(attempts)==failure_count+1 and all(a==attempts[0] for a in attempts)
    assert sleeps==[2,4][:failure_count] and len(got['zero_payload_read_failures'])==failure_count
    assert len(list(tmp_path.glob('download.partial.read-attempts/*.json')))==failure_count


def test_exhaustion_is_fatal_for_fresh_snapshot_and_restart(tmp_path,monkeypatch):
    from pod_versioned_export import classified_export,require_retryable_checkpoint
    t,remote,expected,clock,sleeps=fixture(tmp_path,monkeypatch);calls=[];output=tmp_path/'export';output.mkdir()
    def stream(*a,**kw):calls.append(True);raise transient()
    t.stream=stream
    with pytest.raises(m.SmallReadRecoveryExhausted):
        classified_export(t,'root',output,1100,None,lambda:t.get(expected['path'],tmp_path/'download',expected,1100))
    assert len(calls)==3 and sleeps==[2,4]
    assert not json.loads((output/'failure.json').read_text())['retryable']
    with pytest.raises(EvidenceError,match='fatal'):require_retryable_checkpoint(t,'root',output,1100,None)
    with pytest.raises(EvidenceError,match='preserved partial'):t.get(expected['path'],tmp_path/'download',expected,1100)
    assert len(calls)==3


@pytest.mark.parametrize('fault',['auth','identity','unknown','cleanup','missing-counts','partial','lying-counts','sent','overflow'])
def test_strict_or_nonzero_failures_never_retry(tmp_path,monkeypatch,fault):
    t,remote,expected,clock,sleeps=fixture(tmp_path,monkeypatch);calls=[]
    def stream(argv,dest,*a,**kw):
        calls.append(True);e=transient()
        if fault in ('auth','identity','unknown'):e=EvidenceError(fault)
        elif fault=='cleanup':e.transport_cleanup_diagnostic='synthetic failure'
        elif fault=='missing-counts':del e.transfer_counts
        elif fault in ('partial','lying-counts'):
            dest.write(b'a')
            if fault=='partial':e.transfer_counts['bytes_received']=1
        elif fault=='sent':e.transfer_counts['bytes_sent']=1
        elif fault=='overflow':e.transfer_overflow=b'a'
        raise e
    t.stream=stream
    with pytest.raises(EvidenceError):t.get(expected['path'],tmp_path/'download',expected,1100)
    assert calls==[True] and sleeps==[] and not(tmp_path/'download').exists()


@pytest.mark.parametrize('change',['endpoint','key','hostkey','expected','deadline','rollback'])
def test_retry_cannot_change_selection_or_extend_deadline(tmp_path,monkeypatch,change):
    t,remote,expected,clock,sleeps=fixture(tmp_path,monkeypatch);calls=[]
    if change=='rollback':t.wall=lambda:1000
    def stream(*a,**kw):
        calls.append(True)
        if change=='endpoint':t.profile['host']='127.0.0.2'
        elif change=='key':t.key.write_bytes(b'changed explicit noncredential key')
        elif change=='hostkey':t.known.write_bytes(b'changed synthetic hostkey')
        elif change=='expected':expected['sha256']='b'*64
        else:clock[0]=1101
        raise transient()
    t.stream=stream
    with pytest.raises(EvidenceError):t.get(expected['path'],tmp_path/'download',expected,1100)
    assert len(calls)==1 and sleeps==[] and not(tmp_path/'download').exists()


def test_backoff_cannot_start_another_read_after_expiry(tmp_path,monkeypatch):
    t,remote,expected,clock,sleeps=fixture(tmp_path,monkeypatch);calls=[]
    def stream(*a,**kw):calls.append(True);raise transient()
    def sleep(n):sleeps.append(n);clock[0]=1101
    monkeypatch.setattr(m.time,'sleep',sleep);t.stream=stream
    with pytest.raises(EvidenceError,match='original transfer deadline'):t.get(expected['path'],tmp_path/'download',expected,1100)
    assert calls==[True] and sleeps==[2]


def test_successful_retry_still_rejects_wrong_hash(tmp_path,monkeypatch):
    t,remote,expected,clock,sleeps=fixture(tmp_path,monkeypatch);calls=[]
    def stream(argv,dest,*a,**kw):
        calls.append(True)
        if len(calls)==1:raise transient()
        dest.write(b'bad');return {'bytes_sent':0,'bytes_received':3}
    t.stream=stream
    with pytest.raises(EvidenceError,match='bytes differ'):t.get(expected['path'],tmp_path/'download',expected,1100)
    assert len(calls)==2 and (tmp_path/'download.partial').read_bytes()==b'bad' and not(tmp_path/'download').exists()


def test_actual_failed_child_is_reaped_before_retry(tmp_path,monkeypatch):
    t,remote,calls,processes=setup(tmp_path);(remote/'metadata').write_bytes(b'abc');original=t.popen;attempts=[]
    def popen(command,**kw):
        assert all(p.poll() is not None for p in attempts)
        if not attempts:
            p=subprocess.Popen([sys.executable,'-c','import sys;sys.stderr.write("ssh: connect to host 127.0.0.1 port 2222: Connection timed out\\n");sys.exit(255)'],**kw)
        else:p=original(command,**kw)
        attempts.append(p);return p
    t.popen=popen;monkeypatch.setattr(m.time,'sleep',lambda n:None)
    t.get('metadata',tmp_path/'download',{'path':'metadata','bytes':3,'sha256':sha256(b'abc')},int(time.time())+30)
    assert len(attempts)==2 and all(p.poll() is not None for p in attempts)
    assert (tmp_path/'download').read_bytes()==b'abc'
