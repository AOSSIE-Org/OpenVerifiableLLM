"""Recovery journals stay private while actual artifact inventories remain closed."""
from pathlib import Path
import time
import pytest

import pod_transfer as transfer
import pod_job_client as client
import pod_checkpoint_handoff as handoff
import reconcile_checkpoint_delivery as reconciliation
from test_pod_transfer import setup
from test_pod_checkpoint_handoff import paused,published
from test_pipeline import prepared
from test_workload_health import Clock,JOB,SELECTION
from ovl_pipeline.canonical import EvidenceError,read_json,sha256
from ovl_pipeline.supervision import Journal


def legacy_download(transport,name,destination,expected,deadline,staging,**kwargs):
    """Explicit counterfactual: old direct placement, same real transfer checks."""
    return transport.get(name,destination,expected,deadline,**kwargs)


def one_zero_read_failure(t,monkeypatch):
    original=t.stream;failures=[];deadlines=[]
    def stream(argv,destination,maximum,deadline,**kwargs):
        if argv[2]==transfer.REMOTE_GET and argv[-1]=='get':
            deadlines.append(deadline)
            if not failures:
                failures.append(True)
                error=transfer.TransientTransportError('synthetic zero-byte interruption')
                error.transfer_counts={'bytes_sent':0,'bytes_received':0}
                raise error
        return original(argv,destination,maximum,deadline,**kwargs)
    monkeypatch.setattr(t,'stream',stream)
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    return failures,deadlines


@pytest.mark.parametrize('legacy',[True,False])
@pytest.mark.parametrize('recovery',['ranges','small-read'])
def test_real_direct_export_preserves_journals_without_inventory_exceptions(tmp_path,monkeypatch,legacy,recovery):
    t,remote,calls,processes=setup(tmp_path)
    root=remote/'output';root.mkdir();data=b'exact synthetic payload'*3
    (root/'state').write_bytes(data)
    if legacy:monkeypatch.setattr(transfer,'staged_get',legacy_download)
    if recovery=='ranges':monkeypatch.setattr(transfer,'RANGE_BYTES',16)
    else:failures,deadlines=one_zero_read_failure(t,monkeypatch)
    end=int(time.time())+60;out=tmp_path/'export'
    receipt=client.export_tree(t,'output',out,end)
    assert (out/'files/state').read_bytes()==data
    with Journal(tmp_path/'health-journal').lease() as journal:
        health=Clock().health(journal);health.start_job(SELECTION)
        if legacy:
            with pytest.raises(EvidenceError,match='exact local file tree'):
                health.exported_files(JOB,out/'files',receipt['files'])
        else:
            assert health.exported_files(JOB,out/'files',receipt['files'])
            assert [str(p.relative_to(out/'files')) for p in (out/'files').rglob('*') if p.is_file()]==['state']
            # An unrelated extra file still fails: no broad journal-name exemption.
            (out/'files/foreign.partial.ranges').mkdir()
            (out/'files/foreign.partial.ranges/private.json').write_bytes(b'unselected')
            with pytest.raises(EvidenceError,match='exact local file tree'):
                health.exported_files(JOB,out/'files',receipt['files'])
    receipts=list(out.rglob('attempt-*.json' if recovery=='ranges' else 'failure-*.json'))
    assert receipts and all(read_json(p) for p in receipts)
    if not legacy:assert all(p.is_relative_to(out/'transfers') for p in receipts)
    if recovery=='small-read':assert failures==[True] and len(deadlines)==2 and all(x<=end for x in deadlines)
    assert all(p.poll()==0 for p in processes)


@pytest.mark.parametrize('legacy',[True,False])
def test_primary_checkpoint_handoff_uses_same_actual_ranged_download_fix(prepared,tmp_path,monkeypatch,legacy):
    t,remote,r,root,chain,calls=paused(prepared,tmp_path)
    monkeypatch.setattr(transfer,'SMALL_READ_BYTES',1)
    if legacy:monkeypatch.setattr(transfer,'staged_get',legacy_download)
    out=tmp_path/'snapshot'
    if legacy:
        with pytest.raises(EvidenceError,match='closed safe checkpoint'):
            handoff.snapshot(t,r,root,out,int(time.time())+60)
        assert not(out/'export.json').exists()
    else:
        result=handoff.snapshot(t,r,root,out,int(time.time())+60)
        assert result['result']=='PASS'
        handoff.state_check(r,root,out)
        assert {p.name for p in (out/result['checkpoint_path']).iterdir()}=={'checkpoint.json','state.json','state.safetensors'}
        assert list((out/'transfers').rglob('attempt-*.json'))


def test_uncertain_anchor_reconciliation_retains_recovered_read_outside_anchor_tree(prepared,tmp_path,monkeypatch):
    t,remote,r,root,snapshot,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    handoff.deliver(t,r,root,snapshot,ack,policies,tmp_path/'delivery',int(time.time())+60)
    failures,deadlines=one_zero_read_failure(t,monkeypatch)
    out=tmp_path/'reconcile';end=int(time.time())+60
    result=reconciliation.reconcile(t,r,root,snapshot,ack,policies,out,end)
    assert result['result']=='DELIVERED' and result['remote_writes'] is False
    assert failures==[True] and deadlines and all(x<=end for x in deadlines)
    assert list((out/'transfers').rglob('failure-*.json'))
    assert not list((out/'anchors').rglob('*.read-attempts'))


@pytest.mark.parametrize('fault',['hash','authentication','partial','expired','symlink','competing-destination'])
def test_staged_transfer_never_installs_unverified_or_late_payload(tmp_path,monkeypatch,fault):
    t,remote,calls,processes=setup(tmp_path)
    data=b'correct';(remote/'state').write_bytes(data)
    expected={'path':'state','bytes':len(data),'sha256':sha256(data)}
    destination=tmp_path/'artifact';staging=tmp_path/'private-transfer';end=int(time.time())+60
    clock=[time.time()]
    if fault=='expired':monkeypatch.setattr(t,'wall',lambda:clock[0])
    original=t.get
    if fault=='hash':expected['sha256']='0'*64
    elif fault=='symlink':staging.symlink_to(remote,target_is_directory=True)
    elif fault in ('authentication','partial'):
        def failed(name,target,*args,**kwargs):
            if fault=='partial':Path(str(target)+'.partial').write_bytes(b'cor')
            raise EvidenceError('strict synthetic '+fault+' failure')
        monkeypatch.setattr(t,'get',failed)
    else:
        def changed(*args,**kwargs):
            value=original(*args,**kwargs)
            if fault=='expired':clock[0]=end
            else:destination.write_bytes(b'preserve competing caller')
            return value
        monkeypatch.setattr(t,'get',changed)
    with pytest.raises((EvidenceError,FileExistsError)):
        transfer.staged_get(t,'state',destination,expected,end,staging)
    if fault=='competing-destination':assert destination.read_bytes()==b'preserve competing caller'
    else:assert not destination.exists()
    if fault=='partial':assert (staging/'payload.partial').read_bytes()==b'cor'


def test_staged_range_timeout_keeps_original_deadline_and_partial_evidence(tmp_path,monkeypatch):
    from test_checkpoint_range_recovery import virtual_transport
    t,expected,data,clock,attempts=virtual_transport(tmp_path,monkeypatch,failure=lambda n,o:n==1)
    out=tmp_path/'artifact';staging=tmp_path/'private-transfer';events=[]
    result=transfer.staged_get(t,'state',out,expected,1120,staging,progress=events.append)
    assert out.read_bytes()==data and clock[0]==1098
    assert all(x[2]<=1120 for x in attempts)
    assert result['transferred_payload_bytes']==72 and result['bytes_received']==64
    assert (staging/'payload.partial.ranges/attempt-00000.partial').read_bytes()==data[:8]
    assert max(e['bytes_received'] for e in events)==64


@pytest.mark.parametrize('late_operation',['hash','fsync'])
def test_staging_cannot_renew_monotonic_deadline_after_verified_transfer(tmp_path,monkeypatch,late_operation):
    t,remote,calls,processes=setup(tmp_path)
    (remote/'state').write_bytes(b'correct');expected={'path':'state','bytes':7,'sha256':sha256(b'correct')}
    wall=[1000];mono=[1000];t.wall=lambda:wall[0];t.monotonic=lambda:mono[0]
    original=t.get;completed=[False]
    def get(*args,**kwargs):
        value=original(*args,**kwargs);wall[0]=mono[0]=1090;completed[0]=True;return value
    monkeypatch.setattr(t,'get',get)
    if late_operation=='hash':
        real=transfer.file_hash
        def late(path):
            value=real(path)
            if completed[0] and Path(path).name=='payload':wall[0]=1050;mono[0]=1101
            return value
        monkeypatch.setattr(transfer,'file_hash',late)
    else:
        real=transfer.os.fsync
        def late(fd):
            real(fd)
            if completed[0]:wall[0]=1050;mono[0]=1101
        monkeypatch.setattr(transfer.os,'fsync',late)
    target=tmp_path/'artifact'
    with pytest.raises(EvidenceError,match='original deadline'):
        transfer.staged_get(t,'state',target,expected,1100,tmp_path/'private-transfer')
    if late_operation=='hash':assert not target.exists()
    else:assert target.read_bytes()==b'correct'  # Preserved late bytes, no successful receipt.


@pytest.mark.parametrize('fault',['small-exhausted','range-exhausted','authentication','unknown'])
def test_primary_snapshot_reentry_does_not_reset_fatal_download_attempts(prepared,tmp_path,monkeypatch,fault):
    from test_production_boundary_poll import fixture
    import production_boundary_poll as boundary
    hook,remote,out,health,starts,provider=fixture(prepared,tmp_path,monkeypatch)
    first=hook();t=first.transport;original=t.stream;attempts=[]
    if fault=='range-exhausted':monkeypatch.setattr(transfer,'SMALL_READ_BYTES',1)
    def fail(argv,destination,maximum,deadline,**kwargs):
        if argv[2]==transfer.REMOTE_GET and (argv[-1]=='get' or argv[-3]=='range'):
            attempts.append(deadline)
            if fault in ('authentication','unknown'):raise EvidenceError('strict synthetic '+fault)
            count=1 if fault=='range-exhausted' else 0
            if count:destination.write(b'x')
            error=transfer.TransientTransportError('synthetic interrupted immutable read')
            error.transfer_counts={'bytes_sent':0,'bytes_received':count}
            raise error
        return original(argv,destination,maximum,deadline,**kwargs)
    monkeypatch.setattr(t,'stream',fail);monkeypatch.setattr(transfer.time,'sleep',lambda _:None)
    with pytest.raises(EvidenceError):first.poll()
    count=len(attempts);assert count==(3 if fault.endswith('exhausted') else 1)
    failure=read_json(out/'boundaries/boundary-00000/snapshot/failure.json');assert failure['retryable'] is False
    with pytest.raises(EvidenceError,match='fatal'):hook().poll()
    assert len(attempts)==count and len(set(attempts))==1 and not starts
    assert not(out/'boundaries/boundary-00000/snapshot-retry').exists()


def test_allowed_metadata_retry_cannot_rename_repeated_payload_progress(prepared,tmp_path,monkeypatch):
    from test_production_boundary_poll import fixture
    hook,remote,out,health,starts,provider=fixture(prepared,tmp_path,monkeypatch)
    original=handoff.observe;failed=[];events=[];first=[];intents=[]
    def observe(transport,name,destination,*args,**kwargs):
        if name=='chain.json' and Path(destination).parent.name=='after' and not failed:
            first.extend(events)
            intents.append(read_json(out/'boundaries/boundary-00000/intent.json'))
            failed.append(True);error=transfer.TransientTransportError('synthetic zero-byte metadata interruption')
            error.transfer_counts={'bytes_sent':0,'bytes_received':0};raise error
        return original(transport,name,destination,*args,**kwargs)
    monkeypatch.setattr(handoff,'observe',observe)
    monkeypatch.setattr(health,'bytes',lambda operation,counts,**kwargs:events.append((operation,counts,kwargs)))
    # The bounded publisher now consumes its one eligible retry before returning
    # to the enclosing stage. Repeated bytes must retain the original operation
    # identity, and that recovery cannot renew the boundary deadline.
    assert hook().poll()['index']==0
    assert first and events[:len(first)]==first and events[len(first):2*len(first)]==first
    assert read_json(out/'boundaries/boundary-00000/intent.json')==intents[0]
    assert read_json(out/'boundaries/boundary-00000/snapshot/failure.json')['retryable'] is True
    assert (out/'boundaries/boundary-00000/snapshot-retry/export.json').exists()


@pytest.mark.parametrize('during',['mkdir','get-entry'])
def test_original_staging_ceiling_prevents_new_transport_after_clock_rollback(tmp_path,monkeypatch,during):
    t,remote,calls,processes=setup(tmp_path)
    (remote/'state').write_bytes(b'correct');expected={'path':'state','bytes':7,'sha256':sha256(b'correct')}
    wall=[1000];mono=[1000];t.wall=lambda:wall[0];t.monotonic=lambda:mono[0]
    staging=tmp_path/'private-transfer'
    def expire():wall[0]=1050;mono[0]=1101
    if during=='mkdir':
        original=Path.mkdir
        def mkdir(path,*args,**kwargs):
            value=original(path,*args,**kwargs)
            if path==staging:expire()
            return value
        monkeypatch.setattr(Path,'mkdir',mkdir)
    else:
        original=t.get
        def get(*args,**kwargs):expire();return original(*args,**kwargs)
        monkeypatch.setattr(t,'get',get)
    with pytest.raises(EvidenceError,match='original.*deadline'):
        transfer.staged_get(t,'state',tmp_path/'artifact',expected,1100,staging)
    assert calls==[] and processes==[] and not(tmp_path/'artifact').exists()


def test_primary_snapshot_retained_prefix_and_expired_monotonic_budget_are_fatal(prepared,tmp_path,monkeypatch):
    from test_production_boundary_poll import fixture
    hook,remote,out,health,starts,provider=fixture(prepared,tmp_path,monkeypatch)
    first=hook();t=first.transport;original=t.stream;attempts=[]
    wall=[time.time()];mono=[1000];t.wall=lambda:wall[0];t.monotonic=lambda:mono[0]
    monkeypatch.setattr(transfer,'SMALL_READ_BYTES',1)
    def fail(argv,destination,maximum,deadline,**kwargs):
        if argv[2]==transfer.REMOTE_GET and argv[-3]=='range':
            attempts.append(kwargs['monotonic_deadline'])
            count=8 if len(attempts)==1 else 0
            if count:destination.write(b'prefix08')
            else:mono[0]=kwargs['monotonic_deadline']+1
            error=transfer.TransientTransportError('synthetic interrupted immutable read')
            error.transfer_counts={'bytes_sent':0,'bytes_received':count};raise error
        return original(argv,destination,maximum,deadline,**kwargs)
    monkeypatch.setattr(t,'stream',fail)
    with pytest.raises(transfer.RangeRecoveryExhausted):first.poll()
    assert len(attempts)==2 and attempts[0]==attempts[1]
    snap=out/'boundaries/boundary-00000/snapshot'
    assert read_json(snap/'failure.json')['retryable'] is False
    assert any(p.read_bytes()==b'prefix08' for p in snap.rglob('attempt-00000.partial'))
    with pytest.raises(EvidenceError,match='fatal'):hook().poll()
    assert len(attempts)==2 and not starts and not(snap.parent/'snapshot-retry').exists()
