"""Real subprocess streams and file hashes; explicit local SSH endpoint only."""
from pathlib import Path
import io,sys,time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_bulk_export as m
from pod_job_client import tree,export_tree
from test_pod_transfer import setup
from ovl_pipeline.canonical import EvidenceError,file_hash,read_json,verify_inventory


def many(tmp_path):
    t,remote,calls,processes=setup(tmp_path);data=remote/'output';data.mkdir()
    for i in range(100):(data/f'{i:03d}.bin').write_bytes(bytes([i])*(i*4096))
    (data/'nested').mkdir();(data/'nested/empty').write_bytes(b'')
    return t,remote,calls,data


def test_many_files_preserve_every_byte_with_three_ssh_calls(tmp_path):
    t,remote,calls,data=many(tmp_path);updates=[]
    result=export_tree(t,'output',tmp_path/'export',int(time.time())+30,progress=lambda *args:updates.append(args))
    verify_inventory(tmp_path/'export/files',result['files'])
    assert len(result['files'])==101 and len(calls)==3 and updates
    for p in data.rglob('*'):
        if p.is_file():assert file_hash(p)==file_hash(tmp_path/'export/files'/p.relative_to(data))
    assert not(tmp_path/'export/bulk/stream.partial').exists()
    assert (tmp_path/'export/bulk/transfer.json').exists()


@pytest.mark.parametrize('fault',['content','length','symlink','added-after'])
def test_changed_peer_has_no_export_success_and_preserves_transferred_bytes(tmp_path,fault):
    t,remote,calls,data=many(tmp_path);original=t.stream
    def change(argv,*args,**kwargs):
        if len(argv)>2 and argv[2]==m.REMOTE_BULK:
            if fault=='content':(data/'001.bin').write_bytes(b'x'*4096)
            elif fault=='length':(data/'001.bin').write_bytes(b'larger')
            elif fault=='symlink':
                (data/'001.bin').unlink();(data/'001.bin').symlink_to(data/'002.bin')
            result=original(argv,*args,**kwargs)
            if fault=='added-after':(data/'unlisted').write_bytes(b'new')
            return result
        return original(argv,*args,**kwargs)
    t.stream=change
    with pytest.raises(EvidenceError):export_tree(t,'output',tmp_path/'export',int(time.time())+30)
    assert not(tmp_path/'export/export.json').exists()
    if fault!='added-after':assert (tmp_path/'export/bulk/stream.partial').exists()
    else:assert (tmp_path/'export/files/001.bin').exists()


def test_corrupted_local_stream_is_detected_even_if_peer_exit_is_zero(tmp_path):
    t,remote,calls,data=many(tmp_path);files=tree(t,'output',int(time.time())+30)
    def forged(argv,destination,maximum,deadline,**kwargs):
        destination.write(b'x'*maximum);return {'bytes_received':maximum,'bytes_sent':kwargs['source_bytes'],'process_exit_code':0}
    t.stream=forged
    with pytest.raises(EvidenceError,match='bulk file differs'):m.receive(t,'output',files,tmp_path/'bulk',int(time.time())+30)
    assert (tmp_path/'bulk/stream.partial').exists() and not(tmp_path/'bulk/transfer.json').exists()


@pytest.mark.parametrize('fault',['traversal','duplicate','parent-collision','oversize','whole-root-implicit'])
def test_invalid_selected_inventory_never_launches_ssh(tmp_path,fault):
    t,remote,calls,data=many(tmp_path)
    files=[{'path':'one','bytes':0,'sha256':file_hash(data/'000.bin')}];name='output'
    if fault=='traversal':files[0]['path']='../one'
    elif fault=='duplicate':files*=2
    elif fault=='parent-collision':files.append({**files[0],'path':'one/child'})
    elif fault=='oversize':files[0]['bytes']=2**40+1
    else:name='.'
    with pytest.raises(EvidenceError):m.receive(t,name,files,tmp_path/'bulk',int(time.time())+30)
    assert not calls


def test_explicit_whole_root_and_empty_files_are_retained(tmp_path):
    t,remote,calls,processes=setup(tmp_path);(remote/'empty').write_bytes(b'');(remote/'one').write_bytes(b'one')
    files=tree(t,'.',int(time.time())+30,whole_root=True)
    result=m.receive(t,'.',files,tmp_path/'bulk',int(time.time())+30,whole_root=True)
    verify_inventory(Path(result['files_directory']),files)


def test_versioned_bulk_preserves_snapshots_and_rechecks_reused_objects(tmp_path):
    from pod_versioned_export import export
    t,remote,calls,data=many(tmp_path);deadline=int(time.time())+60
    a=export(t,'output',tmp_path/'store',tmp_path/'first',deadline)
    assert len(calls)==3 and len(a['transfers'])==1
    b=export(t,'output',tmp_path/'store',tmp_path/'second',deadline)
    assert len(calls)==5 and not b['transfers'] and len(b['reused_paths'])==101
    (data/'050.bin').write_bytes(b'new')
    c=export(t,'output',tmp_path/'store',tmp_path/'third',deadline)
    assert len(c['transfers'])==1
    verify_inventory(tmp_path/'first/files',a['files']);verify_inventory(tmp_path/'third/files',c['files'])
    obj=next(p for p in (tmp_path/'store/objects').iterdir() if p.stat().st_size)
    obj.chmod(0o600);obj.write_bytes(b'corrupted')
    with pytest.raises(EvidenceError,match='retained export object differs'):export(t,'output',tmp_path/'store',tmp_path/'fourth',deadline)
    assert not(tmp_path/'fourth/export.json').exists()


def test_nonadjacent_parent_collision_refused_before_ssh(tmp_path):
    t,remote,calls,data=many(tmp_path);files=[{'path':n,'bytes':0,'sha256':file_hash(data/'000.bin')} for n in ['a','a.x','a/b']]
    with pytest.raises(EvidenceError,match='parent collision'):m.receive(t,'output',files,tmp_path/'bulk',int(time.time())+30)
    assert not calls


def test_wheel_plus_names_preserved_in_complete_remote_export(tmp_path):
    t,remote,calls,data=many(tmp_path);(data/'torch-2.14.0+cu130.whl').write_bytes(b'wheel-name-only-fixture')
    r=export_tree(t,'output',tmp_path/'export',int(time.time())+30)
    verify_inventory(tmp_path/'export/files',r['files'])


def test_post_stream_deadline_and_insufficient_space_cannot_pass(tmp_path,monkeypatch):
    from collections import namedtuple
    t,remote,calls,data=many(tmp_path);files=tree(t,'output',int(time.time())+30);calls.clear()
    usage=namedtuple('usage','total used free');monkeypatch.setattr(m.shutil,'disk_usage',lambda _:usage(1000,999,1))
    with pytest.raises(EvidenceError,match='headroom'):m.receive(t,'output',files,tmp_path/'low-space',int(time.time())+30)
    assert not calls
    monkeypatch.undo();clock=[100];original=t.stream
    def expire(*args,**kw):
        # Real stream with its own actual transport deadline; only receiver's
        # clock is advanced for the post-stream check.
        args=list(args);args[3]=int(time.time())+30;r=original(*args,**kw);clock[0]=111;return r
    t.stream=expire
    with pytest.raises(EvidenceError,match='deadline'):m.receive(t,'output',files,tmp_path/'expired',110,wall=lambda:clock[0],monotonic=lambda:clock[0])
    assert (tmp_path/'expired/stream.partial').exists() and not(tmp_path/'expired/transfer.json').exists()


def test_disk_full_during_split_preserves_original_stream(tmp_path,monkeypatch):
    t,remote,calls,data=many(tmp_path);files=tree(t,'output',int(time.time())+30);original=Path.open
    def full(path,*args,**kw):
        if path.name=='001.bin' and path.parent==tmp_path/'bulk/files':raise OSError(28,'injected no space')
        return original(path,*args,**kw)
    monkeypatch.setattr(Path,'open',full)
    with pytest.raises(OSError):m.receive(t,'output',files,tmp_path/'bulk',int(time.time())+30)
    assert (tmp_path/'bulk/stream.partial').stat().st_size==sum(f['bytes'] for f in files)
    assert not(tmp_path/'bulk/transfer.json').exists()
