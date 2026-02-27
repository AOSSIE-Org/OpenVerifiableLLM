import os
import hashlib
import tempfile
import pytest
import json
from pathlib import Path
from openverifiablellm import compute_sha256


def test_correct_sha256_output(tmp_path):
    file = tmp_path / "sample.txt"
    content = "hello wikipedia"
    file.write_text(content, encoding="utf-8")

    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    actual = compute_sha256(str(file))
    assert actual == expected


def test_different_content_different_hash(tmp_path):
    file1 = tmp_path / "content_a.txt"
    file2 = tmp_path / "content_b.txt"

    file1.write_text("Content A", encoding="utf-8")
    file2.write_text("Content B", encoding="utf-8")

    assert compute_sha256(file1) != compute_sha256(file2)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        compute_sha256("non_existent_file.txt")


def test_sha256_chunk_processing(tmp_path):
    file = tmp_path / "large.txt"
    content = "x" * 10000
    file.write_text(content, encoding="utf-8")

    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    actual = compute_sha256(str(file))
    assert actual == expected


def test_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    expected = hashlib.sha256(b"").hexdigest()
    actual = compute_sha256(str(file))
    assert actual == expected


def test_binary_file(tmp_path):
    file = tmp_path / "binary.bin"
    with open(file, "wb") as f:
        f.write(b"\x00\x01\x02\x03" * 100)

    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    expected = sha256.hexdigest()
    actual = compute_sha256(str(file))
    assert actual == expected


def test_large_file_exceeds_memory(tmp_path):
    file = tmp_path / "huge.txt"
    with open(file, "wb") as f:
        for _ in range(1000):
            f.write(b"A" * 8192)

    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    expected = sha256.hexdigest()
    actual = compute_sha256(str(file))
    assert actual == expected


def test_path_object_vs_string(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("test content", encoding="utf-8")

    hash_from_str = compute_sha256(str(file))
    hash_from_path = compute_sha256(file)
    assert hash_from_str == hash_from_path
