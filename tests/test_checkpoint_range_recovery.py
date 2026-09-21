"""Exact immutable range transfers and bounded synthetic transport failures."""
import io
import json
import shlex
import subprocess
import sys
import time

import pytest

from test_pod_transfer import setup
import pod_transfer as m
from ovl_pipeline.canonical import EvidenceError, file_hash, sha256


def selected(remote, data):
    (remote/'state').write_bytes(data)
    return {'path':'state','bytes':len(data),'sha256':sha256(data)}


def test_real_ranges_reassemble_and_hash_complete_large_file(tmp_path):
    transport,remote,calls,processes=setup(tmp_path)
    data=bytes(range(256))*(2*m.RANGE_BYTES//256)+b'last range'
    expected=selected(remote,data)
    events=[]
    result=transport.get('state',tmp_path/'download',expected,int(time.time())+30,progress=events.append)
    assert (tmp_path/'download').read_bytes()==data
    assert len(calls)==3 and all(p.returncode==0 for p in processes)
    assert result['bytes_received']==result['transferred_payload_bytes']==len(data)
    assert [x['offset'] for x in result['range_attempts']]==[0,m.RANGE_BYTES,2*m.RANGE_BYTES]
    assert max(x['bytes_received'] for x in events)==len(data)
    assert all(x['bytes_sent']==0 for x in events)
    assert not list(tmp_path.glob('download.partial.ranges/*.partial'))


@pytest.mark.parametrize('size,ranged', [(16*1024**2,False),(16*1024**2+1,True),(64*1024**2,True)])
def test_larger_range_keeps_original_small_read_threshold(tmp_path,size,ranged):
    transport,remote,calls,processes=setup(tmp_path)
    expected=selected(remote,b'x'*size)
    result=transport.get('state',tmp_path/'download',expected,int(time.time())+30)
    argv=shlex.split(calls[0][-1])
    assert len(calls)==1 and all(p.returncode==0 for p in processes)
    assert ('range' in argv)==ranged
    assert ('range_policy' in result)==ranged
    assert result['sha256']==file_hash(tmp_path/'download')==expected['sha256']
    if ranged:
        assert result['range_policy']['bytes']==64*1024**2
        assert result['range_policy']['payload_idle_seconds']==90
        assert result['range_policy']['maximum_retries']==2


def test_medium_file_partial_payload_still_has_range_recovery(tmp_path):
    transport,remote,calls,processes=setup(tmp_path)
    data=b'correct immutable fixture bytes\0'*(1024**2)
    assert 16*1024**2<len(data)<64*1024**2
    expected=selected(remote,data);original=transport.stream;attempts=[]
    def fail_once(argv,destination,maximum,deadline,**kwargs):
        attempts.append((argv,deadline,kwargs.get('payload_idle_seconds')))
        if len(attempts)==1:
            destination.write(data[:1024**2])
            error=m.TransientTransportError('synthetic connection reset')
            error.transfer_counts={'bytes_sent':0,'bytes_received':1024**2}
            raise error
        return original(argv,destination,maximum,deadline,**kwargs)
    transport.stream=fail_once;deadline=int(time.time())+30
    result=transport.get('state',tmp_path/'download',expected,deadline)
    assert len(attempts)==2 and all(x[0][-3]=='range' and x[1]<=deadline and x[2]==90 for x in attempts)
    assert result['transferred_payload_bytes']==len(data)+1024**2
    assert result['bytes_received']==len(data)
    assert (tmp_path/'download.partial.ranges/attempt-00000.partial').read_bytes()==data[:1024**2]
    assert (tmp_path/'download').read_bytes()==data and all(p.returncode==0 for p in processes)


def test_remote_range_cap_remains_closed_at_64_mib(tmp_path):
    transport,remote,calls,processes=setup(tmp_path);size=64*1024**2+1
    with (remote/'state').open('wb') as f:f.truncate(size)
    destination=io.BytesIO()
    with pytest.raises(EvidenceError,match='process failed'):
        transport.stream(['/usr/bin/python3','-c',m.REMOTE_GET,transport.profile['remote_root'],
                          'state',str(size),'range','0',str(size)],destination,size,int(time.time())+30)
    assert len(calls)==1 and destination.getvalue()==b''
    assert all(p.poll() is not None for p in processes)


def virtual_transport(tmp_path,monkeypatch,*,failure=None):
    transport,remote,calls,processes=setup(tmp_path)
    data=bytes(range(64));expected=selected(remote,data)
    monkeypatch.setattr(m,'RANGE_BYTES',16)
    clock=[1000.0];transport.wall=lambda:clock[0];transport.monotonic=lambda:clock[0]
    attempts=[]
    def stream(argv,destination,maximum,deadline,**kwargs):
        offset=int(argv[-2]);length=int(argv[-1]);attempts.append((offset,length,deadline))
        if failure is not None and failure(len(attempts),offset):
            destination.write(data[offset:offset+8])
            if kwargs.get('progress'):kwargs['progress']({'bytes_sent':0,'bytes_received':8})
            clock[0]+=90
            error=m.TransientTransportError('synthetic connection stalled at range deadline')
            error.transfer_counts={'bytes_sent':0,'bytes_received':8}
            raise error
        destination.write(data[offset:offset+length]);clock[0]+=2
        if kwargs.get('progress'):kwargs['progress']({'bytes_sent':0,'bytes_received':length})
        return {'bytes_sent':0,'bytes_received':length,'process_exit_code':0}
    transport.stream=stream
    return transport,expected,data,clock,attempts


def test_early_range_timeout_recovers_without_renewing_total_deadline(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch,failure=lambda n,o:n==1)
    events=[]
    result=transport.get('state',tmp_path/'download',expected,1120,progress=events.append)
    assert clock[0]==1098 and all(x[2]<=1120 for x in attempts)
    assert (tmp_path/'download').read_bytes()==data
    assert result['transferred_payload_bytes']==72 and result['bytes_received']==64
    assert (tmp_path/'download.partial.ranges/attempt-00000.partial').read_bytes()==data[:8]
    assert len(attempts)==5 and [e['bytes_received'] for e in events]==[8,16,32,48,64]


def test_retries_do_not_multiply_logical_progress(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch,failure=lambda n,o:n in (1,2))
    events=[]
    result=transport.get('state',tmp_path/'download',expected,1300,progress=events.append)
    assert result['transferred_payload_bytes']==80
    assert [x['bytes_received'] for x in events][:3]==[8,8,16]
    assert max(x['bytes_received'] for x in events)==64


def test_range_retries_keep_health_high_water_across_journal_restart(tmp_path,monkeypatch):
    from test_workload_health import Clock,SELECTION
    from ovl_pipeline.supervision import Journal
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch,failure=lambda n,o:n in (1,2))
    c=Clock();path=tmp_path/'journal';operation='c'*64;observed=[]
    with Journal(path).lease() as journal:c.health(journal).start_job(SELECTION)
    def progress(counts):
        c.advance(1)
        with Journal(path).lease() as journal:
            health=c.health(journal)
            observed.append(health.bytes(operation,counts,total=len(data)))
    result=transport.get('state',tmp_path/'download',expected,1300,progress=progress)
    with Journal(path).lease() as journal:
        health=c.health(journal);prior=health.progress
        c.advance(1)
        assert not health.bytes(operation,{'bytes_sent':0,'bytes_received':len(data)},total=len(data))
        assert health.progress==prior and health.transfers[operation]['credited_bytes']==len(data)
    assert observed==[False,False,False,False,False,True]
    assert result['transferred_payload_bytes']==80 and result['bytes_received']==64


@pytest.mark.parametrize('fault',['slow-start','insufficient-bandwidth'])
def test_range_policy_limitations_do_not_grant_success(tmp_path,monkeypatch,fault):
    """A synthetic latency model disproves universal benefit from reconnecting."""
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch)
    def modeled(argv,dest,maximum,deadline,**kwargs):
        ranged=argv[-3]=='range';offset=int(argv[-2]) if ranged else 0
        required=95+maximum/64 if fault=='slow-start' else maximum*20
        # This model produces no payload until completion, so inactivity can
        # expire first. Productive streaming is tested through actual stream().
        available=min(deadline-clock[0],kwargs.get('payload_idle_seconds',deadline-clock[0]));attempts.append((ranged,deadline))
        if required>available:
            clock[0]+=available
            error=m.TransientTransportError('synthetic policy model timeout')
            error.transfer_counts={'bytes_sent':0,'bytes_received':0};raise error
        clock[0]+=required;dest.write(data[offset:offset+maximum])
        return {'bytes_sent':0,'bytes_received':maximum,'process_exit_code':0}
    transport.stream=modeled;baseline=io.BytesIO()
    args=['python','script','root','state','64','get']
    if fault=='slow-start':
        transport.stream(args,baseline,64,1660);assert baseline.getvalue()==data
    else:
        with pytest.raises(m.TransientTransportError):transport.stream(args,baseline,64,1660)
    clock[0]=1000;attempts.clear()
    with pytest.raises(m.RangeRecoveryExhausted):transport.get('state',tmp_path/'download',expected,1660)
    assert len(attempts)==3 and clock[0]==1270 and not(tmp_path/'download').exists()


def test_transient_retry_count_is_finite_and_preserves_every_failure(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch,failure=lambda n,o:True)
    with pytest.raises(m.RangeRecoveryExhausted):transport.get('state',tmp_path/'download',expected,2000)
    assert len(attempts)==3 and not(tmp_path/'download').exists()
    assert len(list(tmp_path.glob('download.partial.ranges/*.partial')))==3


def test_total_deadline_expires_before_further_retry(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch,failure=lambda n,o:True)
    with pytest.raises(m.TransientTransportError):transport.get('state',tmp_path/'download',expected,1090)
    assert len(attempts)==1 and attempts[0][2]==1090
    assert not(tmp_path/'download').exists()


@pytest.mark.parametrize('message',['authentication denied','host identity mismatch','unknown process failure','changed selected bytes'])
def test_strict_failures_get_no_range_retry(tmp_path,monkeypatch,message):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch)
    def fail(*a,**kw):attempts.append(True);raise EvidenceError(message)
    transport.stream=fail
    with pytest.raises(EvidenceError,match=message):transport.get('state',tmp_path/'download',expected,1120)
    assert attempts==[True] and not(tmp_path/'download').exists()


def test_changed_unread_range_is_rejected_by_complete_file_hash(tmp_path,monkeypatch):
    transport,remote,calls,processes=setup(tmp_path)
    monkeypatch.setattr(m,'RANGE_BYTES',16)
    expected=selected(remote,b'a'*32)
    original=transport.stream
    def mutate(argv,*a,**kw):
        if argv[-2:] == ['16','16']:(remote/'state').write_bytes(b'a'*16+b'b'*16)
        return original(argv,*a,**kw)
    transport.stream=mutate
    with pytest.raises(EvidenceError,match='bytes differ'):transport.get('state',tmp_path/'download',expected,int(time.time())+30)
    assert not(tmp_path/'download').exists() and (tmp_path/'download.partial').read_bytes()==b'a'*16+b'b'*16
    assert all(p.poll() is not None for p in processes)


@pytest.mark.parametrize('damage',['short','extra'])
def test_range_framing_failure_is_not_retried(tmp_path,monkeypatch,damage):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch)
    def bad(argv,dest,maximum,deadline,**kw):
        attempts.append(True);dest.write(b'x'*(maximum+(-1 if damage=='short' else 1)))
        return {'bytes_sent':0,'bytes_received':len(dest.getbuffer())}
    transport.stream=bad
    with pytest.raises(EvidenceError,match='range length'):transport.get('state',tmp_path/'download',expected,1120)
    assert attempts==[True] and not(tmp_path/'download').exists()


def test_final_hash_must_finish_within_original_deadline(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch)
    original=m.file_hash
    def delayed(path):
        result=original(path)
        if path.name=='download.partial':clock[0]=1121
        return result
    monkeypatch.setattr(m,'file_hash',delayed)
    with pytest.raises(EvidenceError,match='verification exceeded'):transport.get('state',tmp_path/'download',expected,1120)
    assert not(tmp_path/'download').exists() and (tmp_path/'download.partial').read_bytes()==data


def test_wall_clock_rollback_cannot_renew_range_deadline(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch)
    original=transport.stream
    transport.wall=lambda:1000
    def delayed(*a,**kw):
        result=original(*a,**kw);clock[0]=1121;return result
    transport.stream=delayed
    with pytest.raises(EvidenceError,match='original transfer deadline'):transport.get('state',tmp_path/'download',expected,1120)
    assert len(attempts)==1 and not(tmp_path/'download').exists()


def test_connection_lifetime_stall_breaks_whole_stream_but_bounded_ranges_finish(tmp_path,monkeypatch):
    """Real child IO under a synthetic per-connection stall, not a network claim."""
    transport,remote,calls,processes=setup(tmp_path)
    monkeypatch.setattr(m,'RANGE_BYTES',16*1024)
    data=b'x'*(64*1024);expected=selected(remote,data);original=transport.popen
    def stalled_whole_read(command,**kwargs):
        if command[-1].endswith(' get'):
            p=subprocess.Popen([sys.executable,'-c',
                'import os,time;os.write(1,b"x"*16384);time.sleep(15)'],**kwargs)
            processes.append(p);return p
        return original(command,**kwargs)
    transport.popen=stalled_whole_read
    baseline=io.BytesIO();before=time.monotonic()
    with pytest.raises(m.TransientTransportError):
        transport.stream(['/usr/bin/python3','-c',m.REMOTE_GET,transport.profile['remote_root'],'state',str(len(data)),'get'],baseline,len(data),int(time.time())+2)
    assert len(baseline.getvalue())==16384 and time.monotonic()-before<5
    assert all(p.poll() is not None for p in processes)
    result=transport.get('state',tmp_path/'download',expected,int(time.time())+2)
    assert result['bytes_received']==len(data) and (tmp_path/'download').read_bytes()==data
    assert len(result['range_attempts'])==4 and all(p.poll() is not None for p in processes)


def test_conflicting_failed_range_prefix_is_fatal_even_if_retry_matches_final_hash(tmp_path,monkeypatch):
    transport,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch)
    original=transport.stream
    def inconsistent(argv,dest,maximum,deadline,**kwargs):
        if not attempts:
            attempts.append(True);dest.write(b'badbytes')
            error=m.TransientTransportError('synthetic reset');error.transfer_counts={'bytes_sent':0,'bytes_received':8};raise error
        return original(argv,dest,maximum,deadline,**kwargs)
    transport.stream=inconsistent
    with pytest.raises(EvidenceError,match='conflicting bytes'):transport.get('state',tmp_path/'download',expected,1120)
    assert len(attempts)==2 and not(tmp_path/'download').exists()
    assert (tmp_path/'download.partial.ranges/attempt-00000.partial').read_bytes()==b'badbytes'
    assert (tmp_path/'download.partial.ranges/attempt-00001.partial').read_bytes()==data[:16]


def test_real_child_oversize_counts_all_consumed_payload_and_preserves_it(tmp_path,monkeypatch):
    transport,remote,calls,processes=setup(tmp_path);monkeypatch.setattr(m,'RANGE_BYTES',16)
    expected=selected(remote,b'x'*32)
    def oversized(command,**kwargs):
        calls.append(command);p=subprocess.Popen([sys.executable,'-c','import os;os.write(1,b"x"*17)'],**kwargs);processes.append(p);return p
    transport.popen=oversized
    with pytest.raises(EvidenceError,match='exceeded bound'):transport.get('state',tmp_path/'download',expected,int(time.time())+30)
    receipt=json.loads((tmp_path/'download.partial.ranges/attempt-00000.json').read_text())
    assert receipt['bytes_received']==receipt['saved_bytes']==17
    assert (tmp_path/'download.partial.ranges/attempt-00000.partial').read_bytes()==b'x'*17
    assert len(calls)==1 and all(p.poll() is not None for p in processes) and not(tmp_path/'download').exists()


def test_failed_owned_process_cleanup_never_starts_another_connection(tmp_path,monkeypatch):
    import os,signal
    transport,remote,calls,processes=setup(tmp_path,fault='hang');monkeypatch.setattr(m,'RANGE_BYTES',16)
    expected=selected(remote,b'x'*32);original=os.killpg
    try:
        with monkeypatch.context() as patch:
            def denied(*args):raise PermissionError('synthetic group cleanup denial')
            patch.setattr(m.os,'killpg',denied)
            with pytest.raises(m.TransientTransportError) as caught:
                transport.get('state',tmp_path/'download',expected,int(time.time())+1)
            assert hasattr(caught.value,'transport_cleanup_diagnostic')
            assert len(calls)==1 and not(tmp_path/'download').exists()
    finally:
        for p in processes:
            if p.poll() is None:original(p.pid,signal.SIGTERM);p.wait(timeout=3)
    assert all(p.poll() is not None for p in processes)


@pytest.mark.parametrize('change',['endpoint','key','selection'])
def test_selected_identity_and_content_cannot_change_between_ranges(tmp_path,monkeypatch,change):
    transport,remote,calls,processes=setup(tmp_path);monkeypatch.setattr(m,'RANGE_BYTES',16)
    expected=selected(remote,b'x'*32);original=transport.stream
    def switch(*args,**kwargs):
        value=original(*args,**kwargs)
        if change=='endpoint':transport.profile['host']='127.0.0.2'
        elif change=='key':transport.key.write_bytes(b'another explicit noncredential test key')
        else:expected['sha256']='a'*64
        return value
    transport.stream=switch
    with pytest.raises(EvidenceError,match='selection changed'):
        transport.get('state',tmp_path/'download',expected,int(time.time())+30)
    assert len(calls)==1 and not(tmp_path/'download').exists()
