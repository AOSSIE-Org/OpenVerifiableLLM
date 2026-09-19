"""Real tiny bytes; no prior/prover receipt can populate semantic reuse."""
import os
import shutil
import pytest
from test_pipeline import prepared as original_prepared
from ovl_pipeline import data,production_observation as m
from ovl_pipeline.canonical import EvidenceError,read_json,write_json,inventory


@pytest.fixture
def prepared(original_prepared,tmp_path):
    original,manifest=original_prepared
    target=tmp_path/"private-streams"
    shutil.copytree(original,target)
    return target,manifest


def selection(prepared):
    directory,_=prepared
    path=directory/'wikipedia'
    return path,read_json(path/'stream.json')


def test_each_scope_scans_once_but_rehashes_every_use(prepared,monkeypatch):
    path,manifest=selection(prepared);calls=[];hashes=[]
    monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    original=data.validate_stream;verify=m.verify_inventory
    monkeypatch.setattr(data,'validate_stream',lambda *a:calls.append(a) or original(*a))
    monkeypatch.setattr(m,'verify_inventory',lambda *a:hashes.append(a) or verify(*a))
    @m.checked_stream_scope
    def run():
        assert m.validate_stream(path,manifest)==manifest['targets']
        assert m.validate_stream(path,manifest)==manifest['targets']
        assert m.validate_stream(path,manifest)==manifest['targets']
    run();run()
    assert len(calls)==2 and len(hashes)==6
    assert m._checked_streams.get() is None


@pytest.mark.parametrize('name',['tokens.u16','mask.u8','documents.jsonl'])
def test_same_size_and_mtime_content_change_rejected_on_reuse(prepared,monkeypatch,name):
    path,manifest=selection(prepared);monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    @m.checked_stream_scope
    def run():
        m.validate_stream(path,manifest)
        target=path/name;stat=target.stat();contents=bytearray(target.read_bytes());contents[0]^=1
        target.write_bytes(contents);os.utime(target,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with pytest.raises(EvidenceError,match='hash mismatch'):m.validate_stream(path,manifest)
    run()


def test_rehash_does_not_allow_rewritten_manifest_to_inherit_semantic_success(prepared,monkeypatch):
    path,manifest=selection(prepared);monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    @m.checked_stream_scope
    def run():
        m.validate_stream(path,manifest)
        contents=bytearray((path/'mask.u8').read_bytes());contents[0]=2;(path/'mask.u8').write_bytes(contents)
        altered={**manifest,'files':inventory(path,[v['path'] for v in manifest['files']])}
        with pytest.raises(EvidenceError,match='loss-mask'):m.validate_stream(path,altered)
    run()


def test_failure_leaves_no_reusable_scope_or_success(prepared,monkeypatch):
    path,manifest=selection(prepared);monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    @m.checked_stream_scope
    def bad():
        wrong={**manifest,'index_root':'f'*64}
        with pytest.raises(EvidenceError,match='totals/root'):m.validate_stream(path,wrong)
        assert not m._checked_streams.get()
        m.validate_stream(path,manifest)
        raise EvidenceError('later unrelated failure')
    with pytest.raises(EvidenceError,match='later unrelated'):bad()
    assert m._checked_streams.get() is None
    assert m.checked_stream_scope(m.validate_stream)(path,manifest)==manifest['targets']


def test_post_scan_mutation_cannot_populate_success(prepared,monkeypatch):
    path,manifest=selection(prepared);monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    original=data.validate_stream
    def changed(*a):
        value=original(*a);target=path/'tokens.u16';raw=bytearray(target.read_bytes());raw[0]^=1;target.write_bytes(raw);return value
    monkeypatch.setattr(data,'validate_stream',changed)
    with pytest.raises(EvidenceError,match='hash mismatch'):m.checked_stream_scope(m.validate_stream)(path,manifest)
    assert m._checked_streams.get() is None


def test_nested_scopes_refuse_and_cleanup(prepared):
    path,manifest=selection(prepared)
    @m.checked_stream_scope
    def outer():m.checked_stream_scope(m.validate_stream)(path,manifest)
    with pytest.raises(EvidenceError,match='nested'):outer()
    assert m._checked_streams.get() is None
