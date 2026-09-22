"""Actual local byte reuse and failure preservation behind an SSH substitute."""
from pathlib import Path
import sys,time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_versioned_export as m
from test_pod_transfer import setup
from ovl_pipeline.canonical import EvidenceError,file_hash


def test_new_snapshots_retain_all_files_but_transfer_only_new_content(tmp_path):
    t,remote,calls,processes=setup(tmp_path);data=remote/'record';data.mkdir()
    (data/'state-a').write_bytes(b'unchanged safe state bytes');(data/'state-b').write_bytes(b'unchanged safe state bytes')
    (data/'log').write_bytes(b'first log')
    before=m.export(t,'record',tmp_path/'store',tmp_path/'first',int(time.time())+30)
    assert len(before['transfers'])==2 and before['reused_paths']==['state-b']
    (data/'log').write_bytes(b'second log, first version remains preserved')
    after=m.export(t,'record',tmp_path/'store',tmp_path/'second',int(time.time())+30)
    assert len(after['transfers'])==1 and after['reused_paths']==['state-a','state-b']
    assert (tmp_path/'first/files/log').read_bytes()==b'first log'
    for file in ('state-a','state-b'):
        assert (tmp_path/'first/files'/file).stat().st_ino==(tmp_path/'second/files'/file).stat().st_ino
        assert (tmp_path/'first/files'/file).stat().st_mode&0o777==0o400
    assert len(after['files'])==3 and after['numerical_verification']=='NOT_RUN'


def test_changed_object_or_peer_never_gets_success_receipt(tmp_path):
    t,remote,calls,processes=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'original')
    m.export(t,'record',tmp_path/'store',tmp_path/'first',int(time.time())+30)
    obj=tmp_path/'store/objects'/file_hash(data/'state');obj.chmod(0o600);obj.write_bytes(b'altered!')
    with pytest.raises(EvidenceError,match='retained export object differs'):
        m.export(t,'record',tmp_path/'store',tmp_path/'second',int(time.time())+30)
    assert not(tmp_path/'second/export.json').exists() and obj.read_bytes()==b'altered!'


def test_partial_transfer_survives_fresh_retry_and_is_not_credited(tmp_path):
    t,remote,calls,processes=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'actual complete state')
    original=t.get
    def fail(name,destination,*a,**k):
        destination.with_name(destination.name+'.partial').write_bytes(b'incomplete')
        raise EvidenceError('explicit interrupted transfer')
    t.get=fail
    with pytest.raises(EvidenceError,match='interrupted'):
        m.export(t,'record',tmp_path/'store',tmp_path/'first',int(time.time())+30)
    partial=list((tmp_path/'store/incoming').glob('*/verified.partial'));assert len(partial)==1
    assert not list((tmp_path/'store/objects').iterdir()) and not(tmp_path/'first/export.json').exists()
    t.get=original;receipt=m.export(t,'record',tmp_path/'store',tmp_path/'retry',int(time.time())+30)
    assert receipt['result']=='PASS' and partial[0].read_bytes()==b'incomplete'


def test_changed_remote_inventory_after_transfer_preserves_every_version(tmp_path):
    t,remote,calls,processes=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'original')
    original=t.get
    def changed(*a,**k):
        result=original(*a,**k);(data/'state').write_bytes(b'new version');return result
    t.get=changed
    with pytest.raises(EvidenceError,match='snapshot changed'):
        m.export(t,'record',tmp_path/'store',tmp_path/'first',int(time.time())+30)
    assert (tmp_path/'first/files/state').read_bytes()==b'original' and not(tmp_path/'first/export.json').exists()
    t.get=original;m.export(t,'record',tmp_path/'store',tmp_path/'retry',int(time.time())+30)
    assert (tmp_path/'first/files/state').read_bytes()==b'original'
    assert (tmp_path/'retry/files/state').read_bytes()==b'new version'


@pytest.mark.parametrize('kind',['logical','uncached'])
def test_terminal_bounds_reject_before_any_payload_transfer(tmp_path,kind):
    t,remote,calls,_=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'actual state')
    with pytest.raises(EvidenceError,match=kind+' export bound'):
        m.export(t,'record',tmp_path/'store',tmp_path/'export',int(time.time())+30,
                 maximum_bytes=1 if kind=='logical' else 100,maximum_uncached_bytes=1 if kind=='uncached' else 100)
    assert not list((tmp_path/'store/incoming').glob('*')) and not(tmp_path/'export/export.json').exists()


def test_terminal_bounds_rehash_reused_bytes_and_count_actual_uncached_payload(tmp_path):
    t,remote,calls,_=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'actual state')
    m.export(t,'record',tmp_path/'store',tmp_path/'initial',int(time.time())+30)
    (data/'log').write_bytes(b'new')
    result=m.export(t,'record',tmp_path/'store',tmp_path/'final',int(time.time())+30,maximum_bytes=15,maximum_uncached_bytes=3)
    assert result['bounds']['selected_missing_bytes']==3 and result['reused_paths']==['state']
    assert len(result['files'])==2 and len(result['transfers'])==1


def test_final_local_inventory_crossing_deadline_cannot_publish_success(tmp_path,monkeypatch):
    t,remote,calls,_=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'actual state')
    now=[time.time()];deadline=int(now[0])+30;t.wall=lambda:now[0]
    original=m.inventory
    def late(*args,**kwargs):
        result=original(*args,**kwargs);now[0]=deadline+1;return result
    monkeypatch.setattr(m,'inventory',late)
    with pytest.raises(EvidenceError,match='verification exceeded original deadline'):
        m.export(t,'record',tmp_path/'store',tmp_path/'final',deadline)
    assert not(tmp_path/'final/export.json').exists()
    assert (tmp_path/'final/files/state').read_bytes()==b'actual state'


def test_large_uncached_tree_uses_single_copy_transfer_with_full_inventory(tmp_path,monkeypatch):
    t,remote,calls,_=setup(tmp_path);data=remote/'record';data.mkdir()
    for i in range(48):(data/f'state-{i:02d}').write_bytes(bytes([i])*100)
    # Scale only the transfer threshold; real files, hashes, hardlinks and both
    # inventories are exercised rather than synthesizing a successful receipt.
    monkeypatch.setattr(m,'MAXIMUM_BULK_BYTES',1000)
    import pod_bulk_export
    def forbidden(*a,**k):raise AssertionError('large snapshot must not create a full duplicate stream')
    monkeypatch.setattr(pod_bulk_export,'receive',forbidden)
    free=[8*1024**3+4800];checks=[]
    def capacity(path,needed):
        checks.append(needed)
        if free[0]<8*1024**3+needed:raise EvidenceError('synthetic capacity exhausted')
    monkeypatch.setattr(m,'require_space',capacity)
    original=t.get
    def received(*a,**k):
        value=original(*a,**k);free[0]-=100;return value
    t.get=received
    result=m.export(t,'record',tmp_path/'store',tmp_path/'export',int(time.time())+30,
                    maximum_bytes=4800,maximum_uncached_bytes=4800)
    assert result['result']=='PASS' and len(result['files'])==48 and len(result['transfers'])==48
    assert result['bounds']['selected_missing_bytes']==4800 and max(checks)==4800 and min(checks)<=100
    assert not list((tmp_path/'store').rglob('stream.partial'))


def test_individual_export_low_space_preserves_partial_and_rejects_receipt(tmp_path,monkeypatch):
    t,remote,calls,_=setup(tmp_path);data=remote/'record';data.mkdir();(data/'state').write_bytes(b'0123456789')
    from types import SimpleNamespace
    import local_storage
    free=[8*1024**3+10]
    monkeypatch.setattr(local_storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=free[0]))
    def interrupted(name,destination,item,deadline,**kw):
        destination.with_suffix('.partial').write_bytes(b'01234');free[0]=8*1024**3
        kw['progress']({'bytes_received':5,'bytes_sent':0})
        raise AssertionError('low-space callback must refuse')
    t.get=interrupted
    with pytest.raises(EvidenceError,match='headroom'):
        m.export(t,'record',tmp_path/'store',tmp_path/'export',int(time.time())+30)
    assert list((tmp_path/'store').rglob('*.partial')) and not(tmp_path/'export/export.json').exists()
