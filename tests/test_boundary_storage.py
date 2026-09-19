from pathlib import Path
import os
import pytest
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json,write_json
import consolidate_boundary_storage as m


def fixture(tmp_path):
    out=tmp_path/'publisher';state=out/'boundaries/boundary-00000';pub=out/'publications/boundary-00000'
    snap=state/'snapshot/checkpoint-00000';fresh=pub/'checkpoint/download-first'
    target=fresh/'downloaded';cache=fresh/'transport-cache'
    for path in (snap,target,cache):path.mkdir(parents=True)
    for name,body in [('checkpoint.json',b'marker'),('state.json',b'metadata'),('state.safetensors',b'complete state bytes')]:
        (snap/name).write_bytes(body);(cache/name).write_bytes(body);os.link(cache/name,target/name)
    files=inventory(snap,[p.name for p in snap.iterdir()]);root='a'*64;boundary='b'*64
    download={'result':'PASS','files':files,'revision':'c'*40}
    ack={'registration_sha256':root,'boundary_sha256':boundary,'checkpoint_download':download,
         'checkpoint_archive':{'inventory':files,'revision':'c'*40}}
    result={'index':0,'registration_sha256':root,'boundary_sha256':boundary,'ack_sha256':digest(ack)}
    write_json(pub/'ack.json',ack);write_json(state/'complete.json',result)
    write_json(snap.parent/'export.json',{'boundary_sha256':boundary,'registration_sha256':root,
                                       'files':files,'checkpoint_path':snap.name})
    write_json(fresh/'verification.json',download);write_json(fresh/'intent.json',{'force_download':True})
    return out,result,tmp_path/'store',snap,target,cache


def test_full_bytes_all_paths_remain_and_restart_is_idempotent(tmp_path):
    out,r,store,snap,target,cache=fixture(tmp_path)
    original={p:p.read_bytes() for folder in (snap,target,cache) for p in folder.iterdir()}
    result=m.consolidate(out,r,store)
    assert result['independent_physical_copies'] is False
    for p,data in original.items():
        assert p.read_bytes()==data and p.samefile(store/'objects'/file_hash(p))
    assert m.consolidate(out,r,store)==result


@pytest.mark.parametrize('damage',['snapshot','download','object','ack','missing-receipt','symlink'])
def test_corruption_refuses_before_replacing_any_original(tmp_path,damage):
    out,r,store,snap,target,cache=fixture(tmp_path)
    if damage=='snapshot':(snap/'state.safetensors').write_bytes(b'altered')
    elif damage=='download':(cache/'state.safetensors').write_bytes(b'altered')
    elif damage=='object':
        objects=store/'objects';objects.mkdir(parents=True);(objects/file_hash(snap/'state.safetensors')).write_bytes(b'altered')
    elif damage=='ack':write_json(out/'publications/boundary-00000/ack.json',{})
    elif damage=='missing-receipt':(target.parent/'verification.json').unlink()
    else:(cache/'escape').symlink_to(tmp_path.parent)
    before={p:(p.stat().st_ino,p.read_bytes()) for d in (snap,target) for p in d.iterdir()}
    with pytest.raises((EvidenceError,KeyError)):m.consolidate(out,r,store)
    assert all((p.stat().st_ino,p.read_bytes())==old for p,old in before.items())


def test_interrupted_replacement_preserves_bytes_and_finishes_original_copies(tmp_path,monkeypatch):
    out,r,store,snap,target,cache=fixture(tmp_path);replace=m.os.replace;count=0
    def interrupted(a,b):
        nonlocal count
        count+=1
        if count==3:raise OSError('simulated interruption after target before cache replacement')
        return replace(a,b)
    monkeypatch.setattr(m.os,'replace',interrupted)
    with pytest.raises(OSError):m.consolidate(out,r,store)
    monkeypatch.setattr(m.os,'replace',replace)
    m.consolidate(out,r,store)
    for d in (snap,target,cache):
        for p in d.iterdir():assert p.samefile(store/'objects'/file_hash(p))


def test_real_publisher_fixture_consolidates_after_verified_handoff(prepared,tmp_path,monkeypatch):
    from test_production_boundary_poll import fixture as actual_fixture
    hook,remote,out,h,starts,provider=actual_fixture(prepared,tmp_path,monkeypatch)
    result=hook().poll()
    receipt=m.consolidate(out,result,tmp_path/'objects')
    assert receipt['result']=='PASS' and len(starts)==1 and provider.commits==2

from test_pipeline import prepared
