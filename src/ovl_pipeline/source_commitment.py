"""Restricted public source commitment construction for the approved Actions job.

This signs an operator source assertion. Public metadata checks are not a full
download, reconstruction, training proof, or independent third-party verification.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import os
import re
import subprocess
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .acquisition import USER_AGENT, wikipedia_source
from .anchoring import ISSUER, OWNER_ID, REPOSITORY, REPOSITORY_ID, WORKFLOW, PublisherPolicy
from .canonical import EvidenceError, confined, digest, parse_json, read_json, sha256, write_json
from .preparation import validate_contract, validate_source_metadata

REF = "refs/heads/feat/verifiable-wikipedia-pipeline"
REQUEST_DIRECTORY = "project/source-commitments"
REVISION_SENTINEL = "github-actions-head"
MAX_METADATA = 16 * 1024 * 1024


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def actions_revision(root, environ):
    if (environ.get("GITHUB_REPOSITORY") != REPOSITORY or environ.get("GITHUB_REF") != REF
            or environ.get("GITHUB_EVENT_NAME") != "push"):
        raise EvidenceError("source signing restricted to approved repository branch push")
    sha = git(root, "rev-parse", "HEAD")
    if (environ.get("GITHUB_SHA") != sha or not re.fullmatch(r"[0-9a-f]{40}", sha)
            or git(root, "status", "--porcelain", "--untracked-files=no")):
        raise EvidenceError("checkout revision mismatch or tracked modifications")
    return sha


def select_request(root, environ):
    """Only a newly added request at the signing commit may trigger signing.

    Existing requests are append-only; unrelated pushes never re-sign an attempt.
    Full git history is mandatory (the workflow explicitly fetches it).
    """
    actions_revision(root, environ)
    parents = git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2:
        raise EvidenceError("source signing requires a single-parent commit")
    changes = git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames",
                  "HEAD", "--", REQUEST_DIRECTORY)
    if not changes:
        return None
    lines = changes.splitlines()
    if len(lines) != 1 or not lines[0].startswith("A\t"):
        raise EvidenceError("exactly one new source request allowed; existing requests are immutable")
    name = lines[0].split("\t")[1]
    if not re.fullmatch(REQUEST_DIRECTORY + r"/[a-z0-9][a-z0-9-]+\.json", name):
        raise EvidenceError("invalid source request path")
    if len(git(root, "log", "--format=%H", "HEAD", "--", name).splitlines()) != 1:
        raise EvidenceError("source request identity was previously used")
    return name


def construct_statement(request, source_revision):
    if request.get("source_revision") != REVISION_SENTINEL:
        raise EvidenceError("request must bind the actual signing commit")
    value = copy.deepcopy(request)
    value["source_revision"] = source_revision
    validate_contract(value)
    return value


def _public_url(url):
    u = urlsplit(url)
    hosts = {"dumps.wikimedia.org", "huggingface.co", "cdn-lfs.huggingface.co",
             "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co"}
    if (u.scheme != "https" or u.hostname not in hosts or u.username or u.password
            or u.port not in (None, 443) or u.fragment):
        raise EvidenceError("metadata URL outside public source policy")


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_metadata(url):
    _public_url(url)
    with build_opener(PublicRedirects()).open(
            Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}), timeout=60) as r:
        _public_url(r.url)
        if r.status != 200 or r.headers.get("Content-Encoding", "identity") != "identity":
            raise EvidenceError("unexpected public metadata HTTP response")
        if 'rel="next"' in r.headers.get("Link", ""):
            raise EvidenceError("paginated metadata exceeds this bounded source profile")
        data = r.read(MAX_METADATA + 1)
    if len(data) > MAX_METADATA:
        raise EvidenceError("public metadata exceeds byte bound")
    return data


def tree(repo, revision, prefix, fetch):
    suffix = "/" + prefix if prefix else ""
    data = parse_json(fetch(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}{suffix}?recursive=true&expand=false"))
    if type(data) is not list:
        raise EvidenceError("invalid public repository tree")
    files = [e for e in data if e.get("type") == "file"]
    if len({e["path"] for e in files}) != len(files):
        raise EvidenceError("duplicate public tree path")
    return {e["path"]: e for e in files}


def verify_public_metadata(statement, *, fetch=fetch_metadata):
    """Check current public upstream/archive metadata; fetch all small parents.

    For large raw objects check the public LFS SHA-256 and byte count. A separate
    complete anonymous download is still required and never claimed here.
    """
    spec = validate_contract(statement)
    wiki = statement["wikipedia"]
    observed = parse_json(fetch(f"https://dumps.wikimedia.org/enwiki/{wiki['date']}/dumpstatus.json"))
    if wikipedia_source(observed, wiki["date"]).object() != spec.object():
        raise EvidenceError("live official Wikipedia inventory differs")
    archive = statement["archive"]
    remote = tree(archive["repo"], archive["revision"], archive["prefix"], fetch)
    expected_names = {archive["prefix"] + "/" + e["path"] for e in archive["inventory"]}
    if set(remote) != expected_names:
        raise EvidenceError("public archive file set differs from commitment")
    with tempfile.TemporaryDirectory(prefix="ovl-source-metadata-") as tmp:
        root = Path(tmp)
        raw_names = {"wikipedia/" + spec.filename, *("conversation/" + n for n in statement["conversation"]["splits"].values())}
        for e in archive["inventory"]:
            entry = remote[archive["prefix"] + "/" + e["path"]]
            if entry["size"] != e["bytes"]:
                raise EvidenceError("public archive size mismatch")
            if e["path"] in raw_names:
                if entry.get("lfs", {}).get("oid") != e["sha256"] or entry.get("lfs", {}).get("size") != e["bytes"]:
                    raise EvidenceError("public raw archive LFS identity mismatch")
            else:
                data = fetch(f"https://huggingface.co/datasets/{archive['repo']}/resolve/{archive['revision']}/{archive['prefix']}/{e['path']}")
                if len(data) != e["bytes"] or sha256(data) != e["sha256"]:
                    raise EvidenceError("public archive metadata bytes mismatch")
                path = confined(root, e["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        _, _, conv_receipt = validate_source_metadata(statement, root / "wikipedia", root / "conversation")
        conv = statement["conversation"]
        upstream = tree(conv["repo"], conv["revision"], "", fetch)
        parquet_names = {n for n in upstream if n.startswith("data/") and n.endswith(".parquet")}
        if parquet_names != set(conv["splits"].values()):
            raise EvidenceError("conversation split inventory is incomplete")
        for e in conv_receipt["files"]:
            entry = upstream.get(e["path"], {})
            if entry.get("size") != e["bytes"] or entry.get("oid") != e["upstream_blob_id"]:
                raise EvidenceError("public conversation blob identity mismatch")
            if e["path"].endswith(".parquet") and entry.get("lfs", {}).get("oid") != e["sha256"]:
                raise EvidenceError("public conversation LFS identity mismatch")
    return {"schema": "ovl.source-public-metadata-check.v1", "result": "PASS",
            "statement_sha256": digest(statement), "complete_raw_download": "NOT_RUN",
            "data_reconstruction": "NOT_RUN", "training_replay": "NOT_RUN"}


def generate(root, environ, output):
    revision = actions_revision(root, environ)
    name = select_request(root, environ)
    if name is None:
        raise EvidenceError("no new source request at this commit")
    statement = construct_statement(read_json(confined(root, name)), revision)
    if Path(name).stem != statement["run_id"] + "-" + statement["attempt_id"]:
        raise EvidenceError("request filename must bind run and preparation attempt")
    observed = verify_public_metadata(statement)
    policy = PublisherPolicy("ovl.publisher-policy.v2", REPOSITORY, WORKFLOW, ISSUER, REF,
                             revision, digest(statement), "sigstore-production-tuf",
                             REPOSITORY_ID, OWNER_ID, "github-hosted")
    policy.validate()
    output.mkdir(exist_ok=False)
    write_json(output / "statement.json", statement)
    write_json(output / "public-metadata-check.json", observed)
    write_json(output / "ci-self-check-policy.json", asdict(policy))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--select", action="store_true")
    p.add_argument("--output", type=Path, default=Path("anchor-source"))
    args = p.parse_args()
    try:
        if args.select:
            print("present=" + ("true" if select_request(Path.cwd(), os.environ) else "false"))
        else:
            generate(Path.cwd(), os.environ, args.output)
    except Exception as e:
        p.exit(1, f"source commitment refused: {e}\n")


if __name__ == "__main__":
    main()
