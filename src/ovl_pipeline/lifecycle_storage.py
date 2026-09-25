"""Inventory-only immutable publication and clean-download adapters.

Operational journals, caches and transfer metadata stay outside the payload.
Publication is allowed only after a caller's exact-content privacy review. This
transport never supplies publisher trust, signing or scientific acceptance.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import ExitStack
import hashlib
import os
import re
import shutil
import subprocess
import tempfile

from .canonical import EvidenceError, confined, digest, require_digest, verify_inventory
from .lifecycle import Observation
from .lifecycle_artifacts import check_snapshot, durable_tree


def verify_git_parent(request, revision, cache):
    """Read the actual public Git commit object; date-sorted API lists are not ancestry."""
    Path(cache).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='commit-', dir=cache) as temporary:
        root = Path(temporary)
        env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT') if k in os.environ}
        env.update(HOME=temporary, GIT_CONFIG_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
        def git(*args):
            result = subprocess.run(['git', *args], cwd=root, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=60, check=False)
            if result.returncode:
                raise EvidenceError('anonymous publication ancestry fetch failed')
            return result.stdout
        git('init', '--bare', '--quiet')
        kind = 'datasets/' if request['repo_type'] == 'dataset' else ''
        url = 'https://huggingface.co/' + kind + request['repo_id']
        git('fetch', '--quiet', '--depth=1', '--filter=blob:none', '--no-tags', url, revision)
        raw = git('cat-file', 'commit', revision)
        if hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != revision:
            raise EvidenceError('publication commit hash differs')
        parents = [line[7:].decode('ascii') for line in raw.split(b'\n\n', 1)[0].splitlines() if line.startswith(b'parent ')]
        if parents != [request['parent_revision']]:
            raise EvidenceError('publication parent differs from frozen request')


def validate_request(request):
    if set(request) != {"repo_id", "repo_type", "parent_revision", "files"}:
        raise EvidenceError("invalid publication request")
    if not re.fullmatch(r"AOSSIE/[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", request["repo_id"]):
        raise EvidenceError("publication destination outside configured organization")
    if request["repo_type"] not in ("model", "dataset") or not re.fullmatch("[0-9a-f]{40}", request["parent_revision"]):
        raise EvidenceError("immutable parent revision and supported repository type required")
    files = request["files"]
    if type(files) is not list or not files or len({e["path"] for e in files}) != len(files):
        raise EvidenceError("nonempty unique publication inventory required")
    for e in files:
        if set(e) != {"path", "bytes", "sha256"} or type(e["bytes"]) is not int or e["bytes"] < 0:
            raise EvidenceError("invalid publication inventory entry")
        require_digest(e["sha256"])
        confined(Path.cwd(), e["path"])


def download_files(request, revision, prefix, output, cache, *, download=None):
    """Download exact immutable bytes anonymously; HF sidecars remain in cache."""
    validate_request(request)
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("objects/[0-9a-f]{64}", prefix):
        raise EvidenceError("invalid immutable publication identity")
    output, cache = Path(output).resolve(), Path(cache).resolve()
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise EvidenceError("download cache must be separate from release payload")
    if output.exists():
        raise EvidenceError("clean download requires fresh output")
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    output.mkdir(parents=True)
    for entry in request["files"]:
        fetched = Path(download(repo_id=request["repo_id"], repo_type=request["repo_type"], revision=revision,
                               filename=prefix+"/"+entry["path"], cache_dir=str(cache), token=False,
                               force_download=True))
        target = confined(output, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fetched, target)
    check_snapshot(output, {"files": request["files"]})
    durable_tree(output)
    return {"revision": revision, "prefix": prefix, "files": request["files"],
            "anonymous_download": True, "scope": "transport-byte-identity-only"}


class PublishObjects:
    """A single journal effect: stage → immutable remote prefix → clean download.

    API/read/download functions are injectable for offline protocol tests. The
    caller supplies a previously created repository and verified parent revision;
    this adapter never creates repositories or changes visibility/settings.
    """
    def __init__(self, payload, downloads, cache, publication_check, *, api=None, download=None,
                 parent_check=verify_git_parent):
        self.payload = Path(payload)
        self.downloads, self.cache = Path(downloads), Path(cache)
        self.publication_check = publication_check
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api, self.download = api, download
        self.parent_check = parent_check

    def observe(self, operation, request):
        validate_request(request)
        require_digest(operation)
        info = self.api.repo_info(request["repo_id"], repo_type=request["repo_type"], token=False)
        head = info.sha
        if not re.fullmatch("[0-9a-f]{40}", head):
            raise EvidenceError("provider omitted immutable revision")
        prefix = "objects/"+operation
        commits = self.api.list_repo_commits(request["repo_id"], repo_type=request["repo_type"], revision=head, token=False)
        matched = [c.commit_id for c in commits if c.title == "Publish content "+operation]
        if len(matched) > 1:
            raise EvidenceError("multiple commits claim the same publication operation")
        revision = matched[0] if matched else head
        if not re.fullmatch("[0-9a-f]{40}", revision):
            raise EvidenceError("invalid publication commit identity")
        # A complete listing is required. An absent prefix never proves that an
        # uncertain commit was not accepted; the Journal retains sent status.
        names = self.api.list_repo_files(request["repo_id"], repo_type=request["repo_type"], revision=revision, token=False)
        found = sorted(n for n in names if n.startswith(prefix+"/"))
        if not found:
            if matched:
                raise EvidenceError("publication commit omitted its objects")
            return Observation("absent")
        if not matched:
            raise EvidenceError("remote objects exist without the original operation commit")
        expected = sorted(prefix+"/"+e["path"] for e in request["files"])
        if found != expected:
            raise EvidenceError("incomplete or conflicting immutable remote inventory")
        self.parent_check(request, revision, self.cache)
        return Observation("complete", {"operation": operation, "repo_id": request["repo_id"],
                                       "repo_type": request["repo_type"], "revision": revision,
                                       "prefix": prefix, "inventory_sha256": digest(request["files"])})

    def submit(self, operation, request):
        validate_request(request)
        require_digest(operation)
        check_snapshot(self.payload, {"files": request["files"]})
        # This callback is run immediately before upload and must check exact
        # current bytes. A scanner result alone is not semantic privacy review.
        self.publication_check(self.payload, request["files"])
        check_snapshot(self.payload, {"files": request["files"]})
        from huggingface_hub import CommitOperationAdd
        # Unlinked task-owned spool files freeze the reviewed bytes for the SDK.
        # Other cooperating writers can change the input paths, but cannot reach
        # these files by pathname. Hostile same-UID /proc writers are out of scope.
        self.cache.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            operations = []
            for e in request["files"]:
                frozen = stack.enter_context(tempfile.TemporaryFile(dir=self.cache))
                h, size = hashlib.sha256(), 0
                with confined(self.payload, e['path']).open('rb') as source:
                    while chunk := source.read(2**20):
                        frozen.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                if size != e['bytes'] or h.hexdigest() != e['sha256']:
                    raise EvidenceError('reviewed publication bytes changed before freezing')
                frozen.flush()
                frozen.seek(0)
                operations.append(CommitOperationAdd(path_in_repo='objects/'+operation+'/'+e['path'],
                                                     path_or_fileobj=frozen))
            self.api.create_commit(repo_id=request['repo_id'], repo_type=request['repo_type'], operations=operations,
                                   parent_commit=request['parent_revision'], commit_message='Publish content '+operation)

    def adopt(self, operation, request, result):
        """Validate a completed immutable revision even after unrelated branch changes."""
        revision = result['revision']
        if not re.fullmatch('[0-9a-f]{40}', revision):
            raise EvidenceError('invalid recorded publication revision')
        self.parent_check(request, revision, self.cache)
        names = self.api.list_repo_files(request['repo_id'], repo_type=request['repo_type'], revision=revision, token=False)
        prefix = 'objects/'+operation+'/'
        if sorted(n for n in names if n.startswith(prefix)) != sorted(prefix+e['path'] for e in request['files']):
            raise EvidenceError('immutable publication inventory differs')
        self.validate(operation, request, result)

    def validate(self, operation, request, result):
        validate_request(request)
        if (result.get("operation") != operation or result.get("repo_id") != request["repo_id"]
                or result.get("repo_type") != request["repo_type"] or result.get("prefix") != "objects/"+operation
                or result.get("inventory_sha256") != digest(request["files"])):
            raise EvidenceError("publication identity mismatch")
        # Use a new directory even on recovery; never confuse cache metadata or
        # existence of an earlier partial with a complete anonymous download.
        import uuid
        output = self.downloads / (operation+"-"+uuid.uuid4().hex)
        download_files(request, result["revision"], result["prefix"], output, self.cache, download=self.download)
