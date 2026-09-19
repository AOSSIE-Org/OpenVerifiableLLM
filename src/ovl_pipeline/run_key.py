"""Private local Ed25519 run keys, adopted only against an external public pin.

The secret is a 32-byte seed in an owner-only regular file. It is never returned
in a report. Existing or incomplete directories are never silently replaced.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat

from nacl.signing import SigningKey

from .canonical import EvidenceError, canonical, digest, read_json, require_digest, write_json
from .schema import fields


def _identifier(value):
    if type(value) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value):
        raise EvidenceError("invalid run-key identifier")


def _no_symlinks(directory):
    # No symlink component, including the final component. Never follow a link
    # while opening secret material, even in an otherwise owner-only directory.
    directory = Path(os.path.abspath(directory))
    for path in (directory, *directory.parents):
        if path.is_symlink():raise EvidenceError("private key path contains a symlink")
    return directory


def _private_directory(directory):
    directory = _no_symlinks(directory)
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise EvidenceError("run key directory must be owner-owned mode0700")
    return directory


def descriptor(run_id, public_key):
    _identifier(run_id);require_digest(public_key)
    return {"schema": "ovl.run-key.v1", "run_id": run_id, "algorithm": "Ed25519",
            "public_key": public_key, "purpose": "training-trajectory-signatures"}


def create(directory: Path, run_id):
    _identifier(run_id)
    # The parent must already exist: no implicit creation of public cache paths.
    directory = _no_symlinks(directory)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    directory = _private_directory(directory)
    key = SigningKey.generate()
    value = descriptor(run_id, bytes(key.verify_key).hex())
    fd = os.open(directory / "seed.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(key));f.flush();os.fsync(f.fileno())
        write_json(directory / "public.json", value)
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:os.fsync(dir_fd)
        finally:os.close(dir_fd)
    except BaseException:
        # Retain a partial directory. A retry must not destroy the only key copy.
        raise
    return value


def load(directory: Path, *, run_id, expected_public_key):
    expected = descriptor(run_id, expected_public_key)
    directory = _private_directory(directory)
    public_path = directory / "public.json"
    if public_path.is_symlink() or not public_path.is_file():
        raise EvidenceError("missing regular public run-key descriptor")
    value = read_json(public_path)
    fields(value, "schema run_id algorithm public_key purpose", "run-key descriptor")
    if value != expected:
        raise EvidenceError("run-key descriptor differs from external identity pin")
    fd = os.open(directory / "seed.key", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(f.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size != 32):
            raise EvidenceError("run seed must be an owner-owned mode0600 regular unlinked-copy 32-byte file")
        seed = f.read(33)
    if len(seed) != 32:raise EvidenceError("invalid run seed length")
    key = SigningKey(seed)
    if bytes(key.verify_key).hex() != expected_public_key:
        raise EvidenceError("private run key differs from external public identity pin")
    return key


def main():
    p = argparse.ArgumentParser(description=__doc__);s = p.add_subparsers(dest="action", required=True)
    c = s.add_parser("create");a = s.add_parser("check")
    for sub in (c, a):
        sub.add_argument("--directory", type=Path, required=True)
        sub.add_argument("--run-id", required=True)
    a.add_argument("--expected-public-key", required=True)
    args = p.parse_args()
    if args.action == "create":value = create(args.directory, args.run_id)
    else:
        key = load(args.directory, run_id=args.run_id, expected_public_key=args.expected_public_key)
        value = descriptor(args.run_id, bytes(key.verify_key).hex())
    print(canonical({"result": "PASS", "public": value, "public_sha256": digest(value),
                     "production_authorization": "NOT_RUN"}).decode())


if __name__ == "__main__":main()
