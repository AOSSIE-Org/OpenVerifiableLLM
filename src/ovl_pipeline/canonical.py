"""Canonical evidence primitives. See docs/PIPELINE_FORMAT.md for byte rules."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile

import rfc8785


class EvidenceError(ValueError):
    pass


def canonical(value) -> bytes:
    """RFC 8785, restricted to null/bools/safe integers/Unicode/lists/dicts."""
    def check(v):
        if v is None or type(v) in (bool, str):
            return
        if type(v) is int and abs(v) <= 2**53 - 1:
            return
        if type(v) is list:
            for x in v:
                check(x)
            return
        if type(v) is dict and all(type(k) is str for k in v):
            for x in v.values():
                check(x)
            return
        raise EvidenceError(f"unsupported canonical value: {type(v).__name__}")
    check(value)
    try:
        return rfc8785.dumps(value)
    except (ValueError, UnicodeError) as e:
        raise EvidenceError(str(e)) from e


def parse_json(data: bytes, *, canonical_required=False, limit=16 * 1024 * 1024):
    if len(data) > limit:
        raise EvidenceError("JSON size limit")
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise EvidenceError(f"duplicate key: {k}")
            out[k] = v
        return out
    def bad(x):
        raise EvidenceError(f"unsupported JSON number: {x}")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=bad, parse_float=bad)
        encoded = canonical(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as e:
        raise EvidenceError(str(e)) from e
    if canonical_required and data != encoded:
        raise EvidenceError("noncanonical JSON bytes")
    return value


def read_json(path: Path, *, canonical_required=True, limit=16 * 1024 * 1024):
    # Bound the read itself; checking len after read_bytes would allocate first.
    with Path(path).open("rb") as f:
        data = f.read(limit + 1)
    return parse_json(data, canonical_required=canonical_required, limit=limit)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value) -> str:
    return sha256(canonical(value))


def file_hash(path: Path, *, progress=None) -> str:
    h = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
            count += len(block)
            if progress is not None:progress(count)
    return h.hexdigest()


def require_digest(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EvidenceError("expected lowercase SHA-256")
    return value


def confined(root: Path, name: str) -> Path:
    if type(name) is not str or "\\" in name or "\x00" in name:
        raise EvidenceError("invalid inventory path")
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in (".", "..") for x in name.split("/")) or str(p) != name:
        raise EvidenceError("noncanonical or escaping inventory path")
    root = Path(root).resolve()
    candidate = root.joinpath(*p.parts)
    # Reject all symlinks, including ones pointing inside: inventory semantics
    # are regular files only. Roots are caller-owned, not writable by an attacker.
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceError("symlink in inventory")
    if not candidate.resolve().is_relative_to(root):
        raise EvidenceError("path escape")
    return candidate


def inventory(root: Path, names: list[str]) -> list[dict]:
    if len(set(names)) != len(names):
        raise EvidenceError("duplicate inventory path")
    return [{"path": n, "bytes": confined(root, n).stat().st_size,
             "sha256": file_hash(confined(root, n))} for n in sorted(names)]


def verify_inventory(root: Path, entries: list[dict], *, max_bytes=2**40, progress=None):
    if type(entries) is not list or not entries:
        raise EvidenceError("empty or invalid inventory")
    seen, total = set(), 0
    for index,e in enumerate(entries):
        if type(e) is not dict or set(e) != {"path", "bytes", "sha256"}:
            raise EvidenceError("invalid inventory entry")
        if type(e["bytes"]) is not int or e["bytes"] < 0:
            raise EvidenceError("invalid file length")
        require_digest(e["sha256"])
        p = confined(root, e["path"])
        if e["path"] in seen:
            raise EvidenceError("duplicate inventory path")
        seen.add(e["path"])
        total += e["bytes"]
        if total > max_bytes or not p.is_file() or p.stat().st_size != e["bytes"]:
            raise EvidenceError("missing, oversized or wrong-size artifact")
        actual=(file_hash(p) if progress is None else
                file_hash(p,progress=lambda count:progress(index,count,False)))
        if actual != e["sha256"]:
            raise EvidenceError(f"artifact hash mismatch: {e['path']}")
        if progress is not None:progress(index,e['bytes'],True)
    return entries


def atomic_write(path: Path, data: bytes):
    """Atomic visibility plus fsync. Caller supplies a trusted output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, value):
    atomic_write(path, canonical(value))


class Merkle:
    """Streaming RFC6962-style ordered tree, wrapped with version and leaf count."""
    def __init__(self):
        self.frontier = []
        self.count = 0

    def add(self, data: bytes):
        node = hashlib.sha256(b"\x00" + data).digest()
        level, n = 0, self.count
        while n & 1:
            node = hashlib.sha256(b"\x01" + self.frontier[level] + node).digest()
            self.frontier[level] = None
            level += 1
            n >>= 1
        if level == len(self.frontier):
            self.frontier.append(node)
        else:
            self.frontier[level] = node
        self.count += 1

    def root(self):
        node = None
        for part in self.frontier:
            if part is not None:
                node = part if node is None else hashlib.sha256(b"\x01" + part + node).digest()
        if node is None:
            node = hashlib.sha256(b"").digest()
        return sha256(b"ovl.merkle.v1\x00" + struct.pack(">Q", self.count) + node)
