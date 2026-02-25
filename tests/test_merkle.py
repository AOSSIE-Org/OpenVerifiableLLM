import hashlib
import pytest
from pathlib import Path
from openverifiablellm import utils


# Merkle Root Tests

def test_merkle_root_deterministic(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("hello wikipedia")

    root1 = utils.compute_merkle_root(file, chunk_size=4)
    root2 = utils.compute_merkle_root(file, chunk_size=4)

    assert root1 == root2


def test_merkle_root_changes_when_content_changes(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("content A")

    root1 = utils.compute_merkle_root(file)

    file.write_text("content B")

    root2 = utils.compute_merkle_root(file)

    assert root1 != root2


def test_merkle_root_single_chunk_equals_sha256(tmp_path):
    file = tmp_path / "data.txt"
    content = "small file"
    file.write_text(content)

    # Large chunk size ensures single leaf
    merkle_root = utils.compute_merkle_root(file, chunk_size=10_000)

    expected = hashlib.sha256(content.encode()).hexdigest()

    assert merkle_root == expected


def test_merkle_root_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("")

    root = utils.compute_merkle_root(file)

    expected = hashlib.sha256(b"").hexdigest()

    assert root == expected