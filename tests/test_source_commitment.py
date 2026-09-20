"""Source signing admission tests; synthetic HTTP responses are not public evidence."""
import copy
import subprocess

import pytest

from test_preparation import inputs
from ovl_pipeline.anchoring import REPOSITORY
from ovl_pipeline.canonical import EvidenceError, canonical, read_json, write_json
from ovl_pipeline.source_commitment import (REF, REVISION_SENTINEL, actions_revision,
    construct_statement, select_request, verify_public_metadata)


def public_responses(contract, wiki, conv):
    """Build a closed fake public service response map, with genuine fixture bytes."""
    archive = contract["archive"]
    responses = {}
    rows = []
    raw = {"wikipedia/"+contract["wikipedia"]["spec"]["filename"],
           *("conversation/"+n for n in contract["conversation"]["splits"].values())}
    for e in archive["inventory"]:
        row = {"type":"file", "path":archive["prefix"]+"/"+e["path"], "size":e["bytes"]}
        if e["path"] in raw:
            row["lfs"]={"oid":e["sha256"],"size":e["bytes"]}
        else:
            if e["path"].startswith("wikipedia/"): data=(wiki/e["path"].removeprefix("wikipedia/")).read_bytes()
            elif e["path"].startswith("conversation/"):data=(conv/e["path"].removeprefix("conversation/")).read_bytes()
            else:data=b"raw"
            responses[f"https://huggingface.co/datasets/{archive['repo']}/resolve/{archive['revision']}/{row['path']}"]=data
        rows.append(row)
    archive_url=f"https://huggingface.co/api/datasets/{archive['repo']}/tree/{archive['revision']}/{archive['prefix']}?recursive=true&expand=false"
    responses[archive_url]=canonical(rows)
    c=contract["conversation"]
    upstream_url=f"https://huggingface.co/api/datasets/{c['repo']}/tree/{c['revision']}?recursive=true&expand=false"
    upstream=[]
    for e in read_json(conv/"acquisition.json")["files"]:
        row={"type":"file","path":e["path"],"size":e["bytes"],"oid":e["upstream_blob_id"]}
        if e["path"].endswith(".parquet"):row["lfs"]={"oid":e["sha256"],"size":e["bytes"]}
        upstream.append(row)
    responses[upstream_url]=canonical(upstream)
    responses[f"https://dumps.wikimedia.org/enwiki/{contract['wikipedia']['date']}/dumpstatus.json"]=(wiki/"dumpstatus.json").read_bytes()
    return responses, archive_url, upstream_url


def test_public_metadata_positive_makes_no_reconstruction_claim(inputs):
    value,wiki,conv=inputs
    responses,_,_=public_responses(value,wiki,conv)
    fetched=[]
    def fetch(url):fetched.append(url);return responses[url]
    result=verify_public_metadata(value,fetch=fetch)
    assert result["result"]=="PASS" and result["complete_raw_download"]=="NOT_RUN"
    assert result["training_replay"]=="NOT_RUN" and result["data_reconstruction"]=="NOT_RUN"
    assert not any(url.endswith((".parquet",".bz2")) for url in fetched)


@pytest.mark.parametrize("mutation",["raw-lfs","raw-size","missing-file","extra-file","changed-parent","upstream-lfs","upstream-extra-split","upstream-blob","official"])
def test_public_source_mutations_fail_closed(inputs,mutation):
    value,wiki,conv=inputs
    responses,a,u=public_responses(value,wiki,conv)
    import json
    rows=json.loads(responses[a]); upstream=json.loads(responses[u])
    raw=next(e for e in rows if e["path"].endswith(".bz2"))
    if mutation=="raw-lfs":raw["lfs"]["oid"]="0"*64
    elif mutation=="raw-size":raw["size"]-=1
    elif mutation=="missing-file":rows.pop()
    elif mutation=="extra-file":rows.append({"type":"file","path":"raw/synthetic/unbound","size":1})
    elif mutation=="changed-parent":
        url=next(n for n in responses if n.endswith("/acquisition.json"));responses[url]=b"{}"
    elif mutation=="upstream-lfs":next(e for e in upstream if "lfs" in e)["lfs"]["oid"]="0"*64
    elif mutation=="upstream-extra-split":upstream.append({"type":"file","path":"data/train-00001.parquet","size":100,"oid":"0"*40})
    elif mutation=="upstream-blob":upstream[0]["oid"]="0"*40
    elif mutation=="official":
        url=next(n for n in responses if n.startswith("https://dumps."));status=json.loads(responses[url])
        status["jobs"]["articlesdumprecombine"]["status"]="in-progress";responses[url]=canonical(status)
    responses[a]=canonical(rows);responses[u]=canonical(upstream)
    with pytest.raises(EvidenceError):verify_public_metadata(value,fetch=responses.__getitem__)


def test_request_revision_is_bound_to_actual_commit_and_code(inputs):
    value,_,_=inputs
    with pytest.raises(EvidenceError):construct_statement(value,"1"*40)
    value["source_revision"]=REVISION_SENTINEL
    assert construct_statement(value,"1"*40)["source_revision"]=="1"*40
    assert value["source_revision"]==REVISION_SENTINEL
    value["code"][0]["sha256"]="0"*64
    with pytest.raises(EvidenceError):construct_statement(value,"1"*40)


@pytest.mark.parametrize("mutation",["foreign-repo","unpinned-revision","missing-raw","retention","traversal","duplicate"])
def test_archive_policy_rejects_incomplete_or_ambiguous_contracts(inputs,mutation):
    value,_,_=inputs
    value=copy.deepcopy(value);value["source_revision"]=REVISION_SENTINEL;a=value["archive"]
    if mutation=="foreign-repo":a["repo"]="other/evidence"
    elif mutation=="unpinned-revision":a["revision"]="main"
    elif mutation=="missing-raw":a["inventory"]=[e for e in a["inventory"] if not e["path"].endswith(".bz2")]
    elif mutation=="retention":a["retention_days_target"]=1
    elif mutation=="traversal":a["inventory"][0]["path"]="../escape"
    elif mutation=="duplicate":a["inventory"].append(a["inventory"][0])
    with pytest.raises(EvidenceError):construct_statement(value,"1"*40)


@pytest.fixture
def repo(tmp_path):
    root=tmp_path/"git";root.mkdir()
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),"-c","user.name=Test","-c","user.email=test@example.invalid",*args]).decode().strip()
    git("init","-q");(root/"initial").write_text("initial");git("add",".");git("commit","-qm","initial")
    def commit():git("add",".");git("commit","-qm","change")
    def env():return {"GITHUB_REPOSITORY":REPOSITORY,"GITHUB_REF":REF,"GITHUB_EVENT_NAME":"push","GITHUB_SHA":git("rev-parse","HEAD")}
    return root,git,commit,env


def test_request_is_append_only_and_unrelated_commit_does_not_resign(repo):
    root,git,commit,env=repo
    name="project/source-commitments/run-1-preparation-1.json"
    write_json(root/name,{"request":"synthetic"});commit()
    assert select_request(root,env())==name
    (root/"initial").write_text("unrelated");commit()
    assert select_request(root,env()) is None
    write_json(root/name,{"request":"changed"});commit()
    with pytest.raises(EvidenceError,match="immutable"):select_request(root,env())
    git("rm",name);commit()
    with pytest.raises(EvidenceError):select_request(root,env())
    write_json(root/name,{"request":"reused"});commit()
    with pytest.raises(EvidenceError,match="previously"):select_request(root,env())


def test_wrong_actions_identity_or_dirty_checkout_refused(repo):
    root,_,commit,env=repo
    (root/"initial").write_text("second");commit()
    for key,value in [("GITHUB_REPOSITORY","other/repo"),("GITHUB_REF","refs/heads/main"),
                      ("GITHUB_EVENT_NAME","pull_request"),("GITHUB_SHA","0"*40)]:
        e=env();e[key]=value
        with pytest.raises(EvidenceError):actions_revision(root,e)
    (root/"initial").write_text("dirty")
    with pytest.raises(EvidenceError,match="modifications"):actions_revision(root,env())
