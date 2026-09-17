"""Verified resumable HTTPS acquisition for a preselected public source inventory.

No source authenticity claim is inferred from a locally generated receipt. The
caller must retain and publicly commit the upstream metadata identifying this spec.
"""
from __future__ import annotations

import bz2
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import platform
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from .canonical import EvidenceError, canonical, confined, digest, file_hash, read_json, write_json

USER_AGENT = "OpenVerifiableLLM/0.1 (https://github.com/AOSSIE-Org/OpenVerifiableLLM)"
BLOCK = 4 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    bytes: int
    upstream_checksums: dict[str, str]
    allowed_hosts: tuple[str, ...]
    compression: str = "none"
    max_uncompressed_bytes: int = 2**40

    def validate(self):
        _url(self.url, self.allowed_hosts)
        if "/" in self.filename or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.filename):
            raise EvidenceError("invalid source filename")
        if type(self.bytes) is not int or not 0 < self.bytes <= 2**40:
            raise EvidenceError("invalid expected source length")
        if type(self.max_uncompressed_bytes) is not int or not 0 < self.max_uncompressed_bytes <= 2**50:
            raise EvidenceError("invalid decompression budget")
        if self.compression not in ("none", "bz2") or not self.upstream_checksums:
            raise EvidenceError("unsupported source compression or absent upstream checksum")
        lengths = {"md5": 32, "sha1": 40, "sha256": 64}
        for kind, value in self.upstream_checksums.items():
            if kind not in lengths or not re.fullmatch(f"[0-9a-f]{{{lengths[kind]}}}", value):
                raise EvidenceError("invalid upstream checksum")

    def object(self):
        obj = asdict(self)
        obj["allowed_hosts"] = list(self.allowed_hosts)
        return {"schema": "ovl.download-spec.v1", **obj}


def _url(url, hosts):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise EvidenceError("source URL outside explicit public HTTPS policy")
    if parsed.port not in (None, 443):
        raise EvidenceError("nonstandard public HTTPS port")
    return url


class Redirects(HTTPRedirectHandler):
    def __init__(self, hosts):
        self.hosts, self.chain = hosts, []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _url(newurl, self.hosts)
        self.chain.append({"from": req.full_url, "to": newurl, "status": code})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_bz2(path, limit):
    """Check every concatenated stream, rejecting truncation and trailing garbage."""
    dec, count = None, 0
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(BLOCK), b""):
                pending = chunk
                while pending or (dec is not None and not dec.needs_input):
                    if dec is None:
                        dec = bz2.BZ2Decompressor()
                    out = dec.decompress(pending, max_length=BLOCK)
                    count += len(out)
                    if count > limit:
                        raise EvidenceError("decompression budget exceeded")
                    if dec.eof:
                        pending, dec = dec.unused_data, None
                    else:
                        pending = b""
        if dec is not None:
            raise EvidenceError("truncated bzip2 stream")
    except OSError as e:
        raise EvidenceError("invalid bzip2 stream") from e
    return count


def verify_source(path, spec):
    spec.validate()
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec.bytes:
        raise EvidenceError("source size mismatch")
    hashes = {kind: hashlib.new(kind) for kind in {*spec.upstream_checksums, "sha256"}}
    with path.open("rb") as f:
        for data in iter(lambda: f.read(BLOCK), b""):
            for h in hashes.values():
                h.update(data)
    observed = {k: v.hexdigest() for k, v in hashes.items()}
    if any(observed[k] != v for k, v in spec.upstream_checksums.items()):
        raise EvidenceError("upstream checksum mismatch")
    decompressed = verify_bz2(path, spec.max_uncompressed_bytes) if spec.compression == "bz2" else None
    return {"bytes": spec.bytes, "hashes": observed, "decompressed_bytes": decompressed}


def acquire(spec: Source, directory: Path, *, attempts=3, timeout=60, opener_factory=build_opener):
    """Retain partial bytes and immutable attempt receipts; never overwrite a bad final.

    Returns only after full byte/checksum/decompression verification. For deterministic
    transport tests opener_factory may provide an in-memory response; production uses
    urllib's certificate-verifying default HTTPS handler. No credential input exists.
    """
    spec.validate()
    if not 1 <= attempts <= 10:
        raise EvidenceError("invalid acquisition retry bound")
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = confined(directory, spec.filename + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _acquire_locked(spec, directory, attempts, timeout, opener_factory)


def _acquire_locked(spec, directory, attempts, timeout, opener_factory):
    final = confined(directory, spec.filename)
    partial = confined(directory, spec.filename + ".partial")
    contract = confined(directory, spec.filename + ".source.json")
    spec_root = digest(spec.object())
    if contract.exists():
        if read_json(contract) != spec.object():
            raise EvidenceError("existing partial/final source belongs to another inventory")
    else:
        if partial.exists() or final.exists():
            raise EvidenceError("orphan source bytes without original inventory")
        write_json(contract, spec.object())
    if final.exists():
        return {"schema": "ovl.acquisition-result.v1", "spec_root": spec_root,
                "verified": verify_source(final, spec), "network_performed": False}
    for attempt in range(attempts):
        identifier = uuid.uuid4().hex
        receipt_path = confined(directory, spec.filename + f".receipt-{identifier}.json")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec.bytes:
            raise EvidenceError("partial exceeds expected file size")
        receipt = {"schema": "ovl.acquisition-receipt.v1", "spec_root": spec_root,
                   "requested_url": spec.url, "started_utc_operator": datetime.now(timezone.utc).isoformat(),
                   "client": "Python urllib", "python_version": platform.python_version(),
                   "downloader_code_sha256": file_hash(Path(__file__)), "user_agent": USER_AGENT, "resume_offset": offset,
                   "bytes_received": 0, "result": "NOT_COMPLETED"}
        try:
            if offset < spec.bytes:
                redirects = Redirects(spec.allowed_hosts)
                opener = opener_factory(redirects)
                headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with opener.open(Request(spec.url, headers=headers), timeout=timeout) as response:
                    status = response.status
                    receipt.update(final_url=_url(response.url, spec.allowed_hosts), redirects=redirects.chain,
                                   http_status=status, headers={k: v for k, v in response.headers.items()
                                   if k.lower() in {"content-length", "content-range", "content-type", "etag", "last-modified", "date", "content-encoding"}})
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise EvidenceError("unexpected HTTP content encoding")
                    if status == 206:
                        expected = f"bytes {offset}-{spec.bytes-1}/{spec.bytes}"
                        if not offset or response.headers.get("Content-Range") != expected:
                            raise EvidenceError("incorrect resumed Content-Range")
                        mode = "ab"
                    elif status == 200:
                        # A server may ignore Range. Restart without appending a full body.
                        offset, mode = 0, "wb"
                        receipt["effective_offset"] = 0
                    else:
                        raise EvidenceError("unexpected download HTTP status")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != spec.bytes - offset:
                        raise EvidenceError("HTTP content length differs from inventory")
                    with partial.open(mode) as f:
                        while block := response.read(BLOCK):
                            if f.tell() + len(block) > spec.bytes:
                                raise EvidenceError("download exceeds committed size")
                            f.write(block)
                            receipt["bytes_received"] += len(block)
                        f.flush()
                        os.fsync(f.fileno())
            verified = verify_source(partial, spec)
            receipt.update(result="PASS", verified=verified, finished_utc_operator=datetime.now(timezone.utc).isoformat())
            write_json(receipt_path, receipt)
            os.replace(partial, final)
            dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
            result = {"schema": "ovl.acquisition-result.v1", "spec_root": spec_root, "verified": verified,
                      "network_performed": bool(receipt["bytes_received"]), "receipt": receipt_path.name,
                      "receipt_sha256": file_hash(receipt_path)}
            write_json(confined(directory, spec.filename + ".verified.json"), result)
            return result
        except (OSError, URLError, http.client.HTTPException, EvidenceError, ValueError) as e:
            receipt.update(result="FAIL", error_type=type(e).__name__, error=str(e),
                           finished_utc_operator=datetime.now(timezone.utc).isoformat())
            write_json(receipt_path, receipt)
            # Bad digests, protocol violations and permission errors are not transient.
            if isinstance(e, EvidenceError) or (isinstance(e, HTTPError) and e.code < 500 and e.code != 429) or attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 8))
    raise EvidenceError("acquisition did not complete")


def wikipedia_source(status, date):
    """Select exactly the completed monolithic article dump; never mix split files."""
    if not re.fullmatch(r"[0-9]{8}", date):
        raise EvidenceError("invalid dump date")
    job = status["jobs"]["articlesdumprecombine"]
    name = f"enwiki-{date}-pages-articles.xml.bz2"
    if job["status"] != "done" or set(job["files"]) != {name}:
        raise EvidenceError("expected one completed dated monolithic article dump")
    row = job["files"][name]
    url = f"/enwiki/{date}/{name}"
    if row["url"] != url:
        raise EvidenceError("dump date/file URL mismatch")
    source = Source("https://dumps.wikimedia.org" + url, name, row["size"],
                    {"sha1": row["sha1"], "md5": row["md5"]}, ("dumps.wikimedia.org",), "bz2")
    source.validate()
    return source
