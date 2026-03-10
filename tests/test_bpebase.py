import pytest
from pathlib import Path

from openverifiablellm.tokenizer.bpe_tokenizer import BPETokenizer


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for training."""
    text = (
        "Wikipedia is a free online encyclopedia.\n"
        "It is written collaboratively by volunteers.\n"
        "Anyone can edit Wikipedia articles.\n"
        "Wikipedia was launched on January 15 2001.\n"
        "It is one of the most popular websites in the world.\n"
    ) * 500

    text_file = tmp_path / "sample.txt"
    text_file.write_text(text, encoding="utf-8")
    return text_file


@pytest.fixture
def trained_tokenizer(tmp_path, sample_text_file):
    """Train and return path to trained BPETokenizer."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.train(sample_text_file, tmp_path / "tokenizer")
    return tmp_path / "tokenizer"


# ------------------------------------------------------------------
# Training tests
# ------------------------------------------------------------------

def test_bpe_train_creates_artifacts(tmp_path, sample_text_file):
    """Training should produce vocab.json and merges.txt."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    save_path = tmp_path / "tokenizer"

    tokenizer.train(sample_text_file, save_path)

    assert (save_path / "vocab.json").is_file()
    assert (save_path / "merges.txt").is_file()


def test_bpe_train_creates_save_directory(tmp_path, sample_text_file):
    """train() should create save_path directory if it does not exist."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    save_path = tmp_path / "nested" / "tokenizer" / "dir"

    assert not save_path.exists()

    tokenizer.train(sample_text_file, save_path)

    assert save_path.exists()


def test_bpe_train_raises_file_not_found(tmp_path):
    """train() should raise FileNotFoundError for missing text file."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)

    with pytest.raises(FileNotFoundError, match="Training file not found"):
        tokenizer.train(
            tmp_path / "nonexistent.txt",
            tmp_path / "tokenizer"
        )


def test_bpe_train_raises_if_directory_passed(tmp_path, sample_text_file):
    """train() should raise FileNotFoundError if directory passed as text_file."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)

    with pytest.raises(FileNotFoundError, match="Training file not found"):
        tokenizer.train(tmp_path, tmp_path / "tokenizer")


# ------------------------------------------------------------------
# Encode / Decode tests
# ------------------------------------------------------------------

def test_bpe_encode_returns_list_of_ints(trained_tokenizer):
    """encode() should return a list of integers."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.load(trained_tokenizer)

    ids = tokenizer.encode("hello world")

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_bpe_encode_decode_roundtrip(trained_tokenizer):
    """encode then decode should return original text."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.load(trained_tokenizer)

    text = "Wikipedia is a free online encyclopedia"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded.strip() == text.strip()


def test_bpe_encode_works_after_train(tmp_path, sample_text_file):
    """encode() should work immediately after train() without calling load()."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.train(sample_text_file, tmp_path / "tokenizer")

    ids = tokenizer.encode("hello world")

    assert isinstance(ids, list)
    assert len(ids) > 0


def test_bpe_encode_raises_if_not_loaded():
    """encode() should raise RuntimeError if model not loaded."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)

    with pytest.raises(RuntimeError, match="not loaded"):
        tokenizer.encode("hello world")


def test_bpe_decode_raises_if_not_loaded():
    """decode() should raise RuntimeError if model not loaded."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)

    with pytest.raises(RuntimeError, match="not loaded"):
        tokenizer.decode([1, 2, 3])


# ------------------------------------------------------------------
# Load tests
# ------------------------------------------------------------------

def test_bpe_load_from_disk(trained_tokenizer):
    """load() should successfully restore tokenizer from disk."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.load(trained_tokenizer)

    assert tokenizer._tokenizer is not None


def test_bpe_encode_works_after_load(trained_tokenizer):
    """encode() should work correctly after load()."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.load(trained_tokenizer)

    ids = tokenizer.encode("hello world")

    assert isinstance(ids, list)
    assert len(ids) > 0


def test_bpe_load_raises_if_vocab_missing(tmp_path):
    """load() should raise FileNotFoundError if vocab.json not found."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)

    with pytest.raises(FileNotFoundError, match="vocab.json not found"):
        tokenizer.load(tmp_path)


def test_bpe_load_raises_if_merges_missing(tmp_path):
    """load() should raise FileNotFoundError if merges.txt not found."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)

    # Create vocab.json but not merges.txt
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="merges.txt not found"):
        tokenizer.load(tmp_path)


# ------------------------------------------------------------------
# Artifact path tests
# ------------------------------------------------------------------

def test_bpe_get_vocab_path(tmp_path):
    """get_vocab_path() should return path to vocab.json."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    vocab_path = tokenizer.get_vocab_path(tmp_path)

    assert vocab_path == tmp_path / "vocab.json"


def test_bpe_get_merges_path(tmp_path):
    """get_merges_path() should return path to merges.txt."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    merges_path = tokenizer.get_merges_path(tmp_path)

    assert merges_path == tmp_path / "merges.txt"


# ------------------------------------------------------------------
# Special tokens tests
# ------------------------------------------------------------------

def test_bpe_special_tokens_in_vocabulary(trained_tokenizer):
    """Special tokens should be present in trained vocabulary."""
    tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer.load(trained_tokenizer)

    vocab_path = trained_tokenizer / "vocab.json"
    vocab_content = vocab_path.read_text(encoding="utf-8")

    assert "<s>" in vocab_content
    assert "</s>" in vocab_content
    assert "<unk>" in vocab_content
    assert "<pad>" in vocab_content
    assert "<mask>" in vocab_content


# ------------------------------------------------------------------
# Determinism tests
# ------------------------------------------------------------------

def test_bpe_training_is_deterministic(tmp_path, sample_text_file):
    """Training twice on same data should produce same vocab."""
    save_path_1 = tmp_path / "tokenizer_1"
    save_path_2 = tmp_path / "tokenizer_2"

    tokenizer_1 = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer_1.train(sample_text_file, save_path_1)

    tokenizer_2 = BPETokenizer(vocab_size=1000, min_frequency=2)
    tokenizer_2.train(sample_text_file, save_path_2)

    vocab_1 = (save_path_1 / "vocab.json").read_text(encoding="utf-8")
    vocab_2 = (save_path_2 / "vocab.json").read_text(encoding="utf-8")

    assert vocab_1 == vocab_2


# ------------------------------------------------------------------
# Constructor validation tests
# ------------------------------------------------------------------

def test_bpe_raises_if_vocab_size_zero():
    """BPETokenizer should raise ValueError if vocab_size <= 0."""
    with pytest.raises(ValueError, match="vocab_size must be > 0"):
        BPETokenizer(vocab_size=0, min_frequency=2)


def test_bpe_raises_if_min_frequency_zero():
    """BPETokenizer should raise ValueError if min_frequency <= 0."""
    with pytest.raises(ValueError, match="min_frequency must be > 0"):
        BPETokenizer(vocab_size=1000, min_frequency=0)