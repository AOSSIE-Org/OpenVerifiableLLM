"""Upload a closed raw inventory and fully download/verify its immutable revision.

No repository creation, deletion, history rewriting, paid storage or credential
export. A failed upload retains its intent; inspect/adopt the remote commit before
any retry. Download retries use a fresh directory and preserve failed copies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import os
from pathlib import Path
import re

# Downloads require Xet disabled. Uploads may explicitly opt into resumable chunk
# transport before SDK import; this does not change the committed file bytes.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, constants

from ovl_pipeline.acquisition import Source, verify_source
from ovl_pipeline.canonical import EvidenceError, canonical, confined, digest, file_hash, read_json, require_digest, verify_inventory, write_json
from ovl_pipeline.schema import fields, integer

REPO = "AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence"


def validate_plan(plan):
    fields(plan, "schema repo prefix files wikipedia_spec wikipedia_verified", "raw archive plan")
    if plan["schema"] != "ovl.raw-archive-plan.v1" or plan["repo"] != REPO:
        raise EvidenceError("unapproved raw archive destination/schema")
    if plan["prefix"] != "raw/" + digest(plan["files"]):
        raise EvidenceError("archive prefix must be its complete inventory digest")
    files = plan["files"]
    if type(files) is not list or not 10 <= len(files) <= 32:
        raise EvidenceError("invalid raw archive file count")
    for e in files:
        fields(e,"path bytes sha256","raw archive entry")
        confined(Path("."),e["path"])
        integer(e["bytes"],1,2**40,"raw file length");require_digest(e["sha256"])
        if e["path"] not in ("README.md","LICENSES.md") and not e["path"].startswith(("wikipedia/","conversation/")):
            raise EvidenceError("unexpected raw archive component")
    spec_obj = plan["wikipedia_spec"]
    if spec_obj.get("schema") != "ovl.download-spec.v1":
        raise EvidenceError("invalid raw source spec")
    spec = Source(**{k:v for k,v in spec_obj.items() if k!="schema"});spec.validate()
    match=re.fullmatch(r"enwiki-([0-9]{8})-pages-articles.xml.bz2",spec.filename)
    if (not match or spec.url!=f"https://dumps.wikimedia.org/enwiki/{match[1]}/{spec.filename}"
            or spec.allowed_hosts!=["dumps.wikimedia.org"] or spec.compression!="bz2"
            or set(spec.upstream_checksums)!={"md5","sha1"}):
        raise EvidenceError("raw archive requires the complete official Wikipedia dump")
    names = [e["path"] for e in files]
    if names != sorted(set(names)) or not {"README.md", "LICENSES.md", "wikipedia/"+spec.filename} <= set(names):
        raise EvidenceError("raw inventory must be sorted, unique and include source/notices")
    raw = next(e for e in files if e["path"] == "wikipedia/"+spec.filename)
    if raw["bytes"] != spec.bytes or raw["sha256"] != plan["wikipedia_verified"]["hashes"]["sha256"]:
        raise EvidenceError("raw identity differs from acquisition")
    return spec


def upload(plan_path, staging, output):
    plan = read_json(plan_path);validate_plan(plan)
    if output.exists() or staging.is_symlink():
        raise EvidenceError("fresh upload receipt directory and regular staging root required")
    verify_inventory(staging, plan["files"])
    api = HfApi(endpoint="https://huggingface.co")
    info = api.repo_info(REPO, repo_type="dataset")
    if info.private:
        raise EvidenceError("archive destination must be public")
    parent = info.sha
    names = api.list_repo_files(REPO, repo_type="dataset", revision=parent)
    if any(n == plan["prefix"] or n.startswith(plan["prefix"]+"/") for n in names):
        raise EvidenceError("archive prefix already exists; inspect/adopt existing revision, never overwrite")
    output.mkdir(parents=True, exist_ok=False)
    intent = {"schema":"ovl.raw-upload-intent.v1", "repo":REPO, "parent_revision":parent,
              "prefix":plan["prefix"], "plan_sha256":file_hash(plan_path),
              "files":plan["files"], "operator_started_utc":datetime.now(timezone.utc).isoformat(),
              "publisher_code_sha256":file_hash(Path(__file__)),
              "xet_disabled":constants.HF_HUB_DISABLE_XET,
              "huggingface_hub_version":importlib.metadata.version("huggingface_hub")}
    write_json(output/"intent.json",intent)
    commit = api.create_commit(REPO, repo_type="dataset", parent_commit=parent,
        commit_message="Archive complete Wikipedia and OpenAssistant source inputs with provenance",
        operations=[CommitOperationAdd(path_in_repo=plan["prefix"]+"/"+e["path"],
                    path_or_fileobj=confined(staging,e["path"])) for e in plan["files"]], num_threads=2)
    receipt={"schema":"ovl.raw-upload.v1", "result":"PASS", "repo":REPO,
             "revision":commit.oid,"url":commit.commit_url,"intent_sha256":digest(intent),
             "plan_sha256":file_hash(plan_path),"complete_download_verification":"NOT_RUN",
             "operator_finished_utc":datetime.now(timezone.utc).isoformat()}
    write_json(output/"upload.json",receipt)
    return receipt


def download(plan_path, revision, output):
    plan=read_json(plan_path);spec=validate_plan(plan)
    if not constants.HF_HUB_DISABLE_XET:
        raise EvidenceError("start a fresh process with Xet disabled before SDK import")
    if not re.fullmatch(r"[0-9a-f]{40}",revision) or output.exists():
        raise EvidenceError("immutable revision and fresh download directory required")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"download-intent.json",{"repo":REPO,"revision":revision,
               "plan_sha256":file_hash(plan_path),"token":False,"force_download":True,
               "xet_disabled":True,"operator_started_utc":datetime.now(timezone.utc).isoformat()})
    local=output/"downloaded"
    for e in plan["files"]:
        hf_hub_download(REPO,repo_type="dataset",revision=revision,filename=plan["prefix"]+"/"+e["path"],
                        local_dir=local,cache_dir=output/"fresh-hub-cache",token=False,force_download=True,
                        endpoint="https://huggingface.co")
    root=local/plan["prefix"]
    verify_inventory(root,plan["files"])
    verified=verify_source(confined(root,"wikipedia/"+spec.filename),spec)
    if verified != plan["wikipedia_verified"]:
        raise EvidenceError("complete public raw reconstruction input differs from original acquisition")
    receipt={"schema":"ovl.raw-download-verification.v1","result":"PASS","repo":REPO,
             "revision":revision,"plan_sha256":file_hash(plan_path),"files":plan["files"],
             "total_bytes":sum(e["bytes"] for e in plan["files"]),"wikipedia_verified":verified,
             "operator_finished_utc":datetime.now(timezone.utc).isoformat(),
             "verifier_code_sha256":file_hash(Path(__file__)),
             "huggingface_hub_version":importlib.metadata.version("huggingface_hub"),
             "scope":"operator complete anonymous download and raw integrity; not third-party verification",
             "data_transformation_reconstruction":"NOT_RUN","training_replay":"NOT_RUN"}
    write_json(output/"verification.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["upload","download"])
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--staging",type=Path)
    p.add_argument("--revision")
    a=p.parse_args()
    lock=a.plan.with_suffix(a.plan.suffix+".publication.lock")
    try:
        with lock.open("a+b") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.action=="upload":
                if a.staging is None:raise EvidenceError("--staging required")
                result=upload(a.plan,a.staging,a.output)
            else:
                if a.revision is None:raise EvidenceError("--revision required")
                result=download(a.plan,a.revision,a.output)
        print(canonical(result).decode())
    except Exception as e:
        # Preserve diagnostic exception classes without leaking signed URLs or
        # credentials from nested HTTP errors. The original attempt stays intact.
        chain=[];current=e
        while current is not None and len(chain)<12:
            chain.append(type(current).__module__+"."+type(current).__qualname__)
            current=current.__cause__ or current.__context__
        reason=re.sub(r"https?://[^\s]+", "[remote URL redacted]", str(e))
        print(canonical({"result":"FAIL","reason":reason,"exception_types":chain}).decode())
        return 1
    return 0


if __name__=="__main__":raise SystemExit(main())
