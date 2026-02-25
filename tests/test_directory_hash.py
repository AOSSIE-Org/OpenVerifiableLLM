import pytest
from pathlib import Path
from openverifiablellm.utils import compute_directory_hash
import hashlib

def test_directory_hash_is_deterministic(tmp_path):
    # Create files
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"

    file1.write_text("hello", encoding="utf-8")
    file2.write_text("world", encoding="utf-8")

    hash1 = compute_directory_hash(tmp_path)
    hash2 = compute_directory_hash(tmp_path)

    assert hash1 == hash2


def test_directory_hash_changes_on_content_modification(tmp_path):
    file1 = tmp_path / "a.txt"
    file1.write_text("hello", encoding="utf-8")

    original_hash = compute_directory_hash(tmp_path)

    file1.write_text("modified", encoding="utf-8")

    new_hash = compute_directory_hash(tmp_path)

    assert original_hash != new_hash

def test_directory_hash_changes_on_file_addition(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    original_hash = compute_directory_hash(tmp_path)

    (tmp_path / "b.txt").write_text("world", encoding="utf-8")

    new_hash = compute_directory_hash(tmp_path)

    assert original_hash != new_hash

def test_directory_hash_changes_on_rename(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("content", encoding="utf-8")

    original_hash = compute_directory_hash(tmp_path)

    renamed_path = tmp_path / "renamed.txt"
    file_path.rename(renamed_path)

    new_hash = compute_directory_hash(tmp_path)

    assert original_hash != new_hash

def test_hidden_file_is_ignored(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    original_hash = compute_directory_hash(tmp_path)

    # Add hidden file
    (tmp_path / ".hidden.txt").write_text("secret", encoding="utf-8")

    new_hash = compute_directory_hash(tmp_path)

    assert original_hash == new_hash


def test_hidden_directory_is_ignored(tmp_path):
    # Visible file
    (tmp_path / "a.txt").write_text("visible", encoding="utf-8")

    original_hash = compute_directory_hash(tmp_path)

    # Hidden directory with file inside
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("secret", encoding="utf-8")

    new_hash = compute_directory_hash(tmp_path)

    assert original_hash == new_hash

def test_empty_directory_hash(tmp_path):
    # No files inside
    hash_value = compute_directory_hash(tmp_path)

    # SHA256 of empty input
    assert hash_value == hashlib.sha256().hexdigest()

def test_nonexistent_directory_raises(tmp_path):
    missing_path = tmp_path / "non_existent_dir"

    with pytest.raises(FileNotFoundError):
        compute_directory_hash(missing_path)


def test_nested_directories_and_move(tmp_path):
    subdir1 = tmp_path / "subdir1"
    subdir2 = tmp_path / "subdir2"
    subdir1.mkdir()
    subdir2.mkdir()

    file1 = subdir1 / "a.txt"
    file2 = subdir2 / "b.txt"

    file1.write_text("content1", encoding="utf-8")
    file2.write_text("content2", encoding="utf-8")

    original_hash = compute_directory_hash(tmp_path)

    # Move file1 into subdir2
    file1.rename(subdir2 / "a.txt")

    new_hash = compute_directory_hash(tmp_path)

    assert original_hash != new_hash
