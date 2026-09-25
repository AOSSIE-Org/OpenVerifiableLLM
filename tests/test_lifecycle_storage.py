from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from ovl_pipeline.canonical import EvidenceError, inventory
from ovl_pipeline.lifecycle import Journal, exclusive
from ovl_pipeline.lifecycle_storage import PublishObjects, download_files


class Hub:
    def __init__(self):
        self.head = "0"*40
        self.commits = []
        self.files = {}
        self.trees = {'0'*40: {}}
        self.parents = {'0'*40: None}
        self.calls = 0
        self.lose_response = False

    def repo_info(self, *args, **kwargs):
        assert kwargs["token"] is False
        return SimpleNamespace(sha=self.head)

    def list_repo_commits(self, *args, **kwargs):
        assert kwargs["token"] is False
        revision = kwargs['revision']
        ancestry = []
        while revision is not None:
            ancestry.append(revision)
            revision = self.parents[revision]
        return [c for c in self.commits if c.commit_id in ancestry]

    def list_repo_files(self, *args, **kwargs):
        assert kwargs["token"] is False
        return sorted(self.trees[kwargs['revision']])

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.head
        self.calls += 1
        for operation in kwargs["operations"]:
            with operation.as_file() as stream:
                self.files[operation.path_in_repo] = stream.read()
        parent = self.head
        self.head = "1"*40
        self.parents[self.head] = parent
        self.trees[self.head] = dict(self.files)
        self.commits.insert(0, SimpleNamespace(commit_id=self.head, title=kwargs["commit_message"]))
        if self.lose_response:
            raise ConnectionError("synthetic acknowledgement lost")

    def download(self, **kwargs):
        assert kwargs["token"] is False and kwargs["force_download"] is True
        assert kwargs["revision"] == "1"*40
        cache = Path(kwargs["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        (cache/".transport-metadata").write_text("synthetic cache metadata")
        blob = cache/"blob"
        blob.write_bytes(self.trees[kwargs['revision']][kwargs["filename"]])
        return blob

    def parent_check(self, request, revision, cache):
        if self.parents[revision] != request['parent_revision']:
            raise EvidenceError('publication parent differs from frozen request')


def fixture(tmp_path):
    payload=tmp_path/"payload"
    payload.mkdir()
    (payload/"model.txt").write_text("synthetic model transport fixture")
    request={"repo_id":"AOSSIE/synthetic-lifecycle-fixture","repo_type":"dataset", "parent_revision":"0"*40,
             "files":inventory(payload,["model.txt"])}
    return payload,request


def test_publication_lost_response_and_unrelated_head_change(tmp_path):
    payload,request=fixture(tmp_path)
    hub=Hub()
    hub.lose_response=True
    reviews=[]
    adapter=PublishObjects(payload,tmp_path/"downloads",tmp_path/"cache",lambda p,f:reviews.append(f),
                           api=hub,download=hub.download,parent_check=hub.parent_check)
    with exclusive(tmp_path/"private"):
        journal=Journal(tmp_path/"private",{})
        with pytest.raises(ConnectionError):
            journal.effect("publish",request,adapter,deadline=time.time()+20)
    assert hub.calls==1 and len(reviews)==1
    with exclusive(tmp_path/"private"):
        journal=Journal(tmp_path/"private",{})
        result=journal.effect("publish",request,adapter,deadline=time.time()+20)
        assert result["revision"]=="1"*40
        hub.head="2"*40
        hub.parents[hub.head] = '1'*40
        hub.trees[hub.head] = {**hub.files, 'unrelated.txt': b'unrelated bytes'}
        hub.commits.insert(0,SimpleNamespace(commit_id=hub.head,title="Unrelated content"))
        assert journal.effect("publish",request,adapter,deadline=time.time()+20)==result
        hub.head='0'*40  # Original commit remains available off current history.
        assert journal.effect('publish',request,adapter,deadline=time.time()+20)==result
    assert hub.calls==1
    assert not list((tmp_path/"downloads").rglob(".transport-metadata"))
    assert len(list((tmp_path/"downloads").glob("*/model.txt")))==3


def test_unreviewed_payload_is_never_submitted(tmp_path):
    payload,request=fixture(tmp_path)
    hub=Hub()
    def reject(*_):
        raise EvidenceError("content has not passed publication review")
    adapter=PublishObjects(payload,tmp_path/"downloads",tmp_path/"cache",reject,api=hub)
    with pytest.raises(EvidenceError,match="publication review"):
        adapter.submit("a"*64,request)
    assert hub.calls==0


def test_clean_download_rejects_corruption_and_cache_overlap(tmp_path):
    payload,request=fixture(tmp_path)
    hub=Hub()
    hub.files["objects/"+"a"*64+"/model.txt"]=b"wrong bytes"
    hub.trees['1'*40] = dict(hub.files)
    with pytest.raises(EvidenceError,match="wrong-size|hash"):
        download_files(request,"1"*40,"objects/"+"a"*64,tmp_path/"download",tmp_path/"cache",download=hub.download)
    with pytest.raises(EvidenceError,match="cache"):
        download_files(request,"1"*40,"objects/"+"a"*64,tmp_path/"cache",tmp_path/"cache",download=hub.download)


def test_reviewed_bytes_are_frozen_before_sdk_consumption(tmp_path):
    payload,request=fixture(tmp_path)
    original=(payload/'model.txt').read_bytes()
    hub=Hub()
    create=hub.create_commit
    def race(**kwargs):
        (payload/'model.txt').write_bytes(b'synthetic-unreviewed-sentinel')
        return create(**kwargs)
    hub.create_commit=race
    adapter=PublishObjects(payload,tmp_path/'downloads',tmp_path/'cache',lambda *_:None,api=hub)
    adapter.submit('a'*64,request)
    assert hub.files['objects/'+'a'*64+'/model.txt']==original


def test_matching_payload_with_wrong_parent_is_not_adopted(tmp_path):
    payload,request=fixture(tmp_path)
    hub=Hub()
    adapter=PublishObjects(payload,tmp_path/'downloads',tmp_path/'cache',lambda *_:None,
                           api=hub,download=hub.download,parent_check=hub.parent_check)
    adapter.submit('a'*64, request)
    hub.parents['1'*40]='f'*40
    hub.parents['f'*40]=None
    with pytest.raises(EvidenceError,match='parent differs'):
        adapter.observe('a'*64, request)
