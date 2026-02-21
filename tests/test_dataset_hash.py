import os
import hashlib
import tempfile
import pytest
from openverifiablellm.dataset_hash import compute_sha256


def test_correct_sha256_output(tmp_path):
    # Create a temporary file
    file = tmp_path / "sample.txt"
    content = "hello wikipedia"
    file.write_text(content, encoding="utf-8")

    # Expected hash using standard hashlib
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Hash using your function
    actual = compute_sha256(str(file))

    # Verify correctness
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