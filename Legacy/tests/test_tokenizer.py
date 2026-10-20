import json

import pytest
import sentencepiece as spm

from openverifiablellm.tokenizer import (
    hash_tokenizer_config,
    train_tokenizer,
)
from openverifiablellm.tokenizer.bpe_tokenizer import BPETokenizer
from openverifiablellm.tokenizer.factory import create_tokenizer
from openverifiablellm.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a small sample text file for testing."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        (
            "Wikipedia is a free online encyclopedia.\n"
            "It is written collaboratively by volunteers.\n"
            "Anyone can edit Wikipedia articles.\n"
            "Wikipedia was launched on January 15 2001.\n"
            "It is one of the most popular websites in the world.\n"
        )
        * 100,
        encoding="utf-8",
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


@pytest.fixture
def trained_sentencepiece_tokenizer(tmp_path, sample_text_file):
    """Train a SentencePiece tokenizer on sample text and return the path."""
    tokenizer_path = tmp_path / "spm_tokenizer"

    train_tokenizer(
        text_file=sample_text_file,
        save_path=tokenizer_path,
        tokenizer_type="sentencepiece",
        vocab_size=50,
        min_frequency=2,
    )

    return tokenizer_path


# ---------------------------------------------------------------------
# Positive Tests
# ---------------------------------------------------------------------


def test_train_tokenizer_creates_files(trained_tokenizer):
    """Training should create vocab.json and merges.txt."""
    assert (trained_tokenizer / "vocab.json").is_file()
    assert (trained_tokenizer / "merges.txt").is_file()


def test_train_tokenizer_is_deterministic(tmp_path, sample_text_file):
    """Training twice on same input should produce identical files."""
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
    """Hashing should return expected keys."""
    hashes = hash_tokenizer_config(trained_tokenizer)

    assert "tokenizer_vocab_hash" in hashes
    assert "tokenizer_merges_hash" in hashes
    assert "tokenizer_vocab_size" in hashes


def test_hash_changes_when_vocab_changes(trained_tokenizer):
    """Modifying vocab.json should change its hash."""
    hashes_before = hash_tokenizer_config(trained_tokenizer)

    vocab_path = trained_tokenizer / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    vocab["new_test_token"] = 99999
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    hashes_after = hash_tokenizer_config(trained_tokenizer)

    assert hashes_before["tokenizer_vocab_hash"] != hashes_after["tokenizer_vocab_hash"]


def test_hash_changes_when_merges_change(trained_tokenizer):
    """Modifying merges.txt should change its hash."""
    hashes_before = hash_tokenizer_config(trained_tokenizer)

    merges_path = trained_tokenizer / "merges.txt"
    original = merges_path.read_text(encoding="utf-8")

    merges_path.write_text(original + "\nxx yy", encoding="utf-8")

    hashes_after = hash_tokenizer_config(trained_tokenizer)

    assert hashes_before["tokenizer_merges_hash"] != hashes_after["tokenizer_merges_hash"]


def test_vocab_size_matches_actual(trained_tokenizer):
    """Reported vocab size should match actual vocab.json length."""
    hashes = hash_tokenizer_config(trained_tokenizer)

    vocab_path = trained_tokenizer / "vocab.json"
    actual_size = len(json.loads(vocab_path.read_text(encoding="utf-8")))

    assert hashes["tokenizer_vocab_size"] == actual_size


# ---------------------------------------------------------------------
# Negative Tests (API Hardening)
# ---------------------------------------------------------------------


def test_train_tokenizer_invalid_vocab_size(sample_text_file, tmp_path):
    with pytest.raises(ValueError, match="vocab_size must be > 0"):
        train_tokenizer(
            sample_text_file,
            tmp_path / "tok",
            vocab_size=0,
        )


def test_train_tokenizer_invalid_min_frequency(sample_text_file, tmp_path):
    with pytest.raises(ValueError, match="min_frequency must be > 0"):
        train_tokenizer(
            sample_text_file,
            tmp_path / "tok",
            min_frequency=0,
        )


def test_train_tokenizer_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_tokenizer(
            tmp_path / "does_not_exist.txt",
            tmp_path / "tok",
        )


def test_hash_tokenizer_missing_vocab(tmp_path):
    tokenizer_path = tmp_path / "tok"
    tokenizer_path.mkdir()

    (tokenizer_path / "merges.txt").write_text("dummy", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        hash_tokenizer_config(tokenizer_path)


def test_hash_tokenizer_missing_merges(tmp_path):
    tokenizer_path = tmp_path / "tok"
    tokenizer_path.mkdir()

    (tokenizer_path / "vocab.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        hash_tokenizer_config(tokenizer_path)


# ---------------------------------------------------------------------
# SentencePiece Tests
# ---------------------------------------------------------------------


def test_train_sentencepiece_tokenizer_creates_files(trained_sentencepiece_tokenizer):
    """SentencePiece training should create spm.model and spm.vocab."""
    assert (trained_sentencepiece_tokenizer / "spm.model").is_file()
    assert (trained_sentencepiece_tokenizer / "spm.vocab").is_file()


def test_train_sentencepiece_tokenizer_is_deterministic(tmp_path, sample_text_file):
    """Training twice with SentencePiece on same text should produce identical files."""
    path1 = tmp_path / "spm1"
    path2 = tmp_path / "spm2"

    train_tokenizer(sample_text_file, path1, tokenizer_type="sentencepiece", vocab_size=50)
    train_tokenizer(sample_text_file, path2, tokenizer_type="sentencepiece", vocab_size=50)

    sp1_proc = spm.SentencePieceProcessor(model_file=str(path1 / "spm.model"))
    sp2_proc = spm.SentencePieceProcessor(model_file=str(path2 / "spm.model"))

    assert sp1_proc.get_piece_size() == sp2_proc.get_piece_size()
    pieces1 = [sp1_proc.id_to_piece(i) for i in range(sp1_proc.get_piece_size())]
    pieces2 = [sp2_proc.id_to_piece(i) for i in range(sp2_proc.get_piece_size())]
    assert pieces1 == pieces2

    scores1 = [sp1_proc.get_score(i) for i in range(sp1_proc.get_piece_size())]
    scores2 = [sp2_proc.get_score(i) for i in range(sp2_proc.get_piece_size())]
    assert scores1 == scores2

    vocab1 = (path1 / "spm.vocab").read_text(encoding="utf-8")
    vocab2 = (path2 / "spm.vocab").read_text(encoding="utf-8")
    assert vocab1 == vocab2


def test_hash_tokenizer_config_sentencepiece_returns_hashes(trained_sentencepiece_tokenizer):
    """Hashing SentencePiece tokenizer should return model and vocab hashes."""
    hashes = hash_tokenizer_config(trained_sentencepiece_tokenizer)

    assert hashes["tokenizer_type"] == "sentencepiece"
    assert "tokenizer_vocab_hash" in hashes
    assert "tokenizer_model_hash" in hashes
    assert hashes["tokenizer_merges_hash"] is None
    assert hashes["tokenizer_vocab_size"] > 0


def test_hash_changes_when_sentencepiece_vocab_changes(trained_sentencepiece_tokenizer):
    """Modifying spm.vocab should change its hash."""
    hashes_before = hash_tokenizer_config(trained_sentencepiece_tokenizer)

    vocab_path = trained_sentencepiece_tokenizer / "spm.vocab"
    content = vocab_path.read_text(encoding="utf-8")
    vocab_path.write_text(content + "\nextra_token\t0.0\n", encoding="utf-8")

    hashes_after = hash_tokenizer_config(trained_sentencepiece_tokenizer)

    assert hashes_before["tokenizer_vocab_hash"] != hashes_after["tokenizer_vocab_hash"]
    assert hashes_after["tokenizer_vocab_size"] == hashes_before["tokenizer_vocab_size"] + 1


def test_hash_changes_when_sentencepiece_model_changes(trained_sentencepiece_tokenizer):
    """Modifying spm.model should change model hash."""
    hashes_before = hash_tokenizer_config(trained_sentencepiece_tokenizer)

    model_path = trained_sentencepiece_tokenizer / "spm.model"
    data = bytearray(model_path.read_bytes())
    data[0] ^= 0xFF
    model_path.write_bytes(bytes(data))

    hashes_after = hash_tokenizer_config(trained_sentencepiece_tokenizer)

    assert hashes_before["tokenizer_model_hash"] != hashes_after["tokenizer_model_hash"]


def test_hash_tokenizer_sentencepiece_missing_model(tmp_path):
    tokenizer_path = tmp_path / "spm_missing_model"
    tokenizer_path.mkdir()
    (tokenizer_path / "spm.vocab").write_text("token\t0.0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="spm.model"):
        hash_tokenizer_config(tokenizer_path)


def test_hash_tokenizer_sentencepiece_missing_vocab(tmp_path):
    tokenizer_path = tmp_path / "spm_missing_vocab"
    tokenizer_path.mkdir()
    (tokenizer_path / "spm.model").write_bytes(b"dummy_model_bytes")

    with pytest.raises(FileNotFoundError, match="spm.vocab"):
        hash_tokenizer_config(tokenizer_path)


def test_hash_tokenizer_unsupported_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        hash_tokenizer_config(tmp_path, tokenizer_type="unknown_type")


# ---------------------------------------------------------------------
# create_tokenizer Factory Tests
# ---------------------------------------------------------------------


def test_create_tokenizer_bpe():
    tok = create_tokenizer("bpe", vocab_size=500, min_frequency=2)
    assert isinstance(tok, BPETokenizer)
    assert tok.vocab_size == 500
    assert tok.min_frequency == 2


def test_create_tokenizer_sentencepiece():
    tok = create_tokenizer("sentencepiece", vocab_size=500, min_frequency=2)
    assert isinstance(tok, SentencePieceTokenizer)
    assert tok.vocab_size == 500
    assert tok.min_frequency == 2


def test_create_tokenizer_case_insensitive():
    assert isinstance(create_tokenizer("BPE", vocab_size=100, min_frequency=1), BPETokenizer)
    assert isinstance(create_tokenizer("SentencePiece", vocab_size=100, min_frequency=1), SentencePieceTokenizer)
    assert isinstance(create_tokenizer("sEnTeNcEpIeCe", vocab_size=100, min_frequency=1), SentencePieceTokenizer)


def test_create_tokenizer_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported tokenizer: invalid_type"):
        create_tokenizer("invalid_type", vocab_size=100, min_frequency=1)

