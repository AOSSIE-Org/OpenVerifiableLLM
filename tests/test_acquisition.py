import bz2
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl_pipeline.acquisition import Source, acquire, verify_bz2, verify_source
from ovl_pipeline.canonical import EvidenceError, write_json


def source(data, compression="none"):
    return Source("https://example.org/wiki.bz2", "wiki.bz2", len(data),
                  {"sha1": hashlib.sha1(data).hexdigest()}, ("example.org",), compression)


def factory(data, status=200, headers=None, seen=None):
    class Response(io.BytesIO):
        url = "https://example.org/wiki.bz2"
    class Opener:
        def open(self, request, timeout):
            if seen is not None:
                seen.append(dict(request.header_items()))
            r = Response(data);r.status=status;r.headers=headers or {"Content-Length": str(len(data))}
            return r
    return lambda redirects: Opener()


def test_download_checksums_receipt_and_verified_reuse(tmp_path):
    data = bz2.compress(b"hello " * 10000)
    spec = source(data, "bz2")
    result = acquire(spec, tmp_path, opener_factory=factory(data))
    assert result["verified"]["hashes"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["verified"]["decompressed_bytes"] == 60000
    assert result["network_performed"]
    reused = acquire(spec, tmp_path, opener_factory=lambda _: pytest.fail("unexpected network on reuse"))
    assert not reused["network_performed"]
    assert not (tmp_path / "wiki.bz2.partial").exists()


def test_resume_with_correct_range_and_range_ignored(tmp_path):
    data = b"abcdefghij"
    for status in (200, 206):
        directory = tmp_path / str(status);directory.mkdir()
        spec = source(data)
        write_json(directory / "wiki.bz2.source.json", spec.object())
        (directory / "wiki.bz2.partial").write_bytes(data[:4])
        seen = []
        response = data if status == 200 else data[4:]
        headers = {"Content-Length": str(len(response))}
        if status == 206:
            headers["Content-Range"] = "bytes 4-9/10"
        acquire(spec, directory, opener_factory=factory(response, status, headers, seen))
        assert seen[0]["Range"] == "bytes=4-"
        assert (directory / "wiki.bz2").read_bytes() == data


def test_bad_range_or_checksum_never_promotes(tmp_path):
    data = b"abcdefghij";spec = source(data)
    write_json(tmp_path / "wiki.bz2.source.json", spec.object())
    (tmp_path / "wiki.bz2.partial").write_bytes(data[:4])
    with pytest.raises(EvidenceError, match="Range"):
        acquire(spec, tmp_path, opener_factory=factory(data[4:], 206, {"Content-Range": "bytes 5-9/10"}))
    assert not (tmp_path / "wiki.bz2").exists()
    with pytest.raises(EvidenceError, match="checksum"):
        acquire(spec, tmp_path, opener_factory=factory(b"XXXXXXXXXX"))
    assert not (tmp_path / "wiki.bz2").exists()
    assert len(list(tmp_path.glob("*.receipt-*.json"))) == 2


def test_bz2_all_members_truncation_garbage_and_limit(tmp_path):
    good = bz2.compress(b"A" * 30) + bz2.compress(b"B" * 40)
    p = tmp_path / "source";p.write_bytes(good)
    assert verify_bz2(p, 70) == 70
    with pytest.raises(EvidenceError, match="budget"):
        verify_bz2(p, 69)
    for bad in (good[:-1], good + b"garbage"):
        p.write_bytes(bad)
        with pytest.raises(EvidenceError):
            verify_bz2(p, 100)


def test_inventory_swap_and_unsafe_urls_fail(tmp_path):
    spec=source(b"abc")
    acquire(spec, tmp_path, opener_factory=factory(b"abc"))
    with pytest.raises(EvidenceError, match="another inventory"):
        acquire(replace(spec, url="https://example.org/another.bz2"), tmp_path)
    for url in ("http://example.org/wiki.bz2", "https://user:secret@example.org/wiki.bz2", "https://elsewhere.org/wiki.bz2", "https://example.org/wiki.bz2?token=secret"):
        with pytest.raises(EvidenceError):
            replace(spec, url=url).validate()
