import json
import pytest
from pathlib import Path
from openverifiablellm.tokenizer import (
    train_tokenizer,
    hash_tokenizer_config,
)


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a small sample text file for testing."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "Wikipedia is a free online encyclopedia.\n"
        "It is written collaboratively by volunteers.\n"
        "Anyone can edit Wikipedia articles.\n"
        "Wikipedia was launched on January 15 2001.\n"
        "It is one of the most popular websites in the world.\n" * 100,
        encoding="utf-8"
    )
    return text_file


@pytest.fixture
def trained_tokenizer(tmp_path, sample_text_file):
    """Train a tokenizer on sample text and return the path."""
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(
        text_file=sample_text_file,
        save_path=tokenizer_path,
        vocab_size=1000,
        min_frequency=2,
    )
    return tokenizer_path


def test_train_tokenizer_creates_files(trained_tokenizer):
    """Test that training creates vocab.json and merges.txt."""
    assert (trained_tokenizer / "vocab.json").exists()
    assert (trained_tokenizer / "merges.txt").exists()


def test_train_tokenizer_is_deterministic(tmp_path, sample_text_file):
    """Test that training twice produces identical files."""
    path1 = tmp_path / "tokenizer1"
    path2 = tmp_path / "tokenizer2"

    train_tokenizer(sample_text_file, path1, vocab_size=1000)
    train_tokenizer(sample_text_file, path2, vocab_size=1000)

    vocab1 = (path1 / "vocab.json").read_text(encoding="utf-8")
    vocab2 = (path2 / "vocab.json").read_text(encoding="utf-8")
    assert vocab1 == vocab2

    merges1 = (path1 / "merges.txt").read_text(encoding="utf-8")
    merges2 = (path2 / "merges.txt").read_text(encoding="utf-8")
    assert merges1 == merges2


def test_hash_tokenizer_config_returns_hashes(trained_tokenizer):
    """Test that hashing returns correct keys."""
    hashes = hash_tokenizer_config(trained_tokenizer)
    assert "tokenizer_vocab_hash" in hashes
    assert "tokenizer_merges_hash" in hashes
    assert "tokenizer_vocab_size" in hashes


def test_hash_tokenizer_config_is_deterministic(trained_tokenizer):
    """Test that hashing the same tokenizer twice gives identical hashes."""
    hashes1 = hash_tokenizer_config(trained_tokenizer)
    hashes2 = hash_tokenizer_config(trained_tokenizer)
    assert hashes1["tokenizer_vocab_hash"] == hashes2["tokenizer_vocab_hash"]
    assert hashes1["tokenizer_merges_hash"] == hashes2["tokenizer_merges_hash"]


def test_hash_changes_when_vocab_changes(trained_tokenizer):
    """Test that modifying vocab.json changes its hash."""
    hashes_before = hash_tokenizer_config(trained_tokenizer)

    vocab_path = trained_tokenizer / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab["new_test_token"] = 99999
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    hashes_after = hash_tokenizer_config(trained_tokenizer)
    assert hashes_before["tokenizer_vocab_hash"] != hashes_after["tokenizer_vocab_hash"]


def test_vocab_size_matches_actual(trained_tokenizer):
    """Test that reported vocab size matches actual vocab.json size."""
    hashes = hash_tokenizer_config(trained_tokenizer)
    vocab_path = trained_tokenizer / "vocab.json"
    actual_size = len(json.loads(vocab_path.read_text(encoding="utf-8")))
    assert hashes["tokenizer_vocab_size"] == actual_size
