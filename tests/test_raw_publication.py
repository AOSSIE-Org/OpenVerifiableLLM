"""Raw publication transport contracts with fake SDK calls; no public evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_preparation import inputs
from ovl_pipeline.canonical import EvidenceError, digest, read_json, write_json

SOURCE=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("raw_publication",SOURCE/"scripts/publish_raw_archive.py")
publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)


@pytest.fixture
def archive(inputs,tmp_path,monkeypatch):
    value,wiki,conv=inputs
    stage=tmp_path/"stage";stage.mkdir()
    for e in value["archive"]["inventory"]:
        if e["path"].startswith("wikipedia/"):data=(wiki/e["path"].removeprefix("wikipedia/")).read_bytes()
        elif e["path"].startswith("conversation/"):data=(conv/e["path"].removeprefix("conversation/")).read_bytes()
        else:data=b"raw"
        p=stage/e["path"];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    files=value["archive"]["inventory"]
    plan={"schema":"ovl.raw-archive-plan.v1","repo":publication.REPO,"prefix":"raw/"+digest(files),
          "files":files,"wikipedia_spec":value["wikipedia"]["spec"],
          "wikipedia_verified":read_json(wiki/(value["wikipedia"]["spec"]["filename"]+".verified.json"))["verified"]}
    path=tmp_path/"plan.json";write_json(path,plan)
    monkeypatch.setattr(publication.constants,"HF_HUB_DISABLE_XET",True)
    return plan,path,stage


def test_upload_is_closed_public_inventory_and_parent_commit_bound(archive,tmp_path,monkeypatch):
    plan,path,stage=archive;calls=[]
    class Api:
        def __init__(self,**kwargs):assert kwargs=={"endpoint":"https://huggingface.co"}
        def repo_info(self,*args,**kwargs):return SimpleNamespace(private=False,sha="1"*40)
        def list_repo_files(self,*args,**kwargs):assert kwargs["revision"]=="1"*40;return ["existing-evidence.json"]
        def create_commit(self,*args,**kwargs):
            assert kwargs["parent_commit"]=="1"*40
            assert all(o.path_in_repo.startswith(plan["prefix"]+"/") for o in kwargs["operations"])
            calls.append(kwargs);return SimpleNamespace(oid="2"*40,commit_url="synthetic-only")
    monkeypatch.setattr(publication,"HfApi",Api)
    result=publication.upload(path,stage,tmp_path/"upload")
    assert len(calls)==1 and len(calls[0]["operations"])==len(plan["files"])
    assert result["complete_download_verification"]=="NOT_RUN"
    assert read_json(tmp_path/"upload/intent.json")["parent_revision"]=="1"*40


@pytest.mark.parametrize("reason",["private","existing-prefix","local-corruption"])
def test_upload_refuses_overwrite_private_destination_and_changed_bytes(archive,tmp_path,monkeypatch,reason):
    plan,path,stage=archive
    class Api:
        def __init__(self,**kwargs):pass
        def repo_info(self,*args,**kwargs):return SimpleNamespace(private=reason=="private",sha="1"*40)
        def list_repo_files(self,*args,**kwargs):return [plan["prefix"]+"/README.md"]
        def create_commit(self,*args,**kwargs):pytest.fail("mutation should be refused before upload")
    monkeypatch.setattr(publication,"HfApi",Api)
    if reason=="local-corruption":(stage/"README.md").write_text("corrupt")
    with pytest.raises(EvidenceError):publication.upload(path,stage,tmp_path/"upload")
    assert not (tmp_path/"upload").exists()


@pytest.mark.parametrize("corrupt",[False,True])
def test_download_is_anonymous_fresh_and_checks_every_file(archive,tmp_path,monkeypatch,corrupt):
    plan,path,stage=archive;calls=[]
    def fetch(repo,**kwargs):
        assert repo==publication.REPO and kwargs["revision"]=="2"*40
        assert kwargs["token"] is False and kwargs["force_download"] is True
        assert kwargs["endpoint"]=="https://huggingface.co"
        name=kwargs["filename"].removeprefix(plan["prefix"]+"/")
        p=kwargs["local_dir"]/kwargs["filename"];p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b"bad" if corrupt and name=="README.md" else (stage/name).read_bytes());calls.append(name)
        return str(p)
    monkeypatch.setattr(publication,"hf_hub_download",fetch)
    if corrupt:
        with pytest.raises(EvidenceError):publication.download(path,"2"*40,tmp_path/"download")
        assert not (tmp_path/"download/verification.json").exists()
    else:
        result=publication.download(path,"2"*40,tmp_path/"download")
        assert result["result"]=="PASS" and result["training_replay"]=="NOT_RUN"
        assert result["total_bytes"]==sum(e["bytes"] for e in plan["files"])
    assert set(calls)=={e["path"] for e in plan["files"]}
    with pytest.raises(EvidenceError):publication.download(path,"2"*40,tmp_path/"download")


def test_unpinned_revision_or_path_escape_cannot_download(archive,tmp_path):
    plan,path,_=archive
    with pytest.raises(EvidenceError):publication.download(path,"main",tmp_path/"download")
    plan["files"][0]["path"]="../../escape";plan["prefix"]="raw/"+digest(plan["files"])
    write_json(path,plan)
    with pytest.raises(EvidenceError):publication.download(path,"2"*40,tmp_path/"download")
    assert not (tmp_path/"download").exists()
