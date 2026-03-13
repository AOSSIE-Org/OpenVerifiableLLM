import pytest

from openverifiablellm.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


@pytest.fixture
def sample_text_file(tmp_path):
    text = (
        "Wikipedia is a free online encyclopedia.\n"
        "It is written collaboratively by volunteers.\n"
        "Anyone can edit Wikipedia articles.\n"
        "Wikipedia was launched on January 15 2001.\n"
        "It is one of the most popular websites in the world.\n"
    ) * 500

    text_file = tmp_path / "sample.txt"
    text_file.write_text(text)
    return text_file


@pytest.fixture
def trained_tokenizer(tmp_path, sample_text_file):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer.train(sample_text_file, tmp_path / "tokenizer")
    return tmp_path / "tokenizer"


def test_spm_train_creates_artifacts(tmp_path, sample_text_file):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    save_path = tmp_path / "tokenizer"

    tokenizer.train(sample_text_file, save_path)

    assert (save_path / "spm.model").is_file()
    assert (save_path / "spm.vocab").is_file()


def test_spm_train_creates_save_directory(tmp_path, sample_text_file):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    save_path = tmp_path / "nested" / "tokenizer" / "dir"

    assert not save_path.exists()

    tokenizer.train(sample_text_file, save_path)

    assert save_path.exists()


def test_spm_train_raises_file_not_found(tmp_path):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)

    with pytest.raises(FileNotFoundError, match="Training file not found"):
        tokenizer.train(tmp_path / "nonexistent.txt", tmp_path / "tokenizer")


def test_spm_train_raises_if_directory_passed(tmp_path, sample_text_file):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)

    with pytest.raises(FileNotFoundError, match="Training file not found"):
        tokenizer.train(tmp_path, tmp_path / "tokenizer")


def test_spm_train_raises_if_min_frequency_not_one(tmp_path, sample_text_file):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=2)

    with pytest.raises(NotImplementedError, match="min_frequency=2 is not supported"):
        tokenizer.train(sample_text_file, tmp_path / "tokenizer")


def test_spm_encode_returns_list_of_ints(trained_tokenizer):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer.load(trained_tokenizer)

    ids = tokenizer.encode("hello world")

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_spm_encode_decode_roundtrip(trained_tokenizer):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer.load(trained_tokenizer)

    text = "Wikipedia is a free online encyclopedia"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded.strip() == text.strip()


def test_spm_encode_raises_if_not_loaded():
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)

    with pytest.raises(RuntimeError, match="not loaded"):
        tokenizer.encode("hello world")


def test_spm_decode_raises_if_not_loaded():
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)

    with pytest.raises(RuntimeError, match="not loaded"):
        tokenizer.decode([1, 2, 3])


def test_spm_load_from_disk(trained_tokenizer):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer.load(trained_tokenizer)

    assert tokenizer._model is not None


def test_spm_encode_works_after_load(trained_tokenizer):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer.load(trained_tokenizer)

    ids = tokenizer.encode("hello world")

    assert isinstance(ids, list)
    assert len(ids) > 0


def test_spm_load_raises_if_model_missing(tmp_path):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)

    with pytest.raises(FileNotFoundError, match="SentencePiece model not found"):
        tokenizer.load(tmp_path)


def test_spm_get_vocab_path(tmp_path):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    vocab_path = tokenizer.get_vocab_path(tmp_path)

    assert vocab_path == tmp_path / "spm.vocab"


def test_spm_get_merges_path_returns_model_path(tmp_path):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    merges_path = tokenizer.get_merges_path(tmp_path)

    assert merges_path == tmp_path / "spm.model"


def test_spm_special_tokens_in_vocabulary(trained_tokenizer):
    tokenizer = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer.load(trained_tokenizer)

    vocab_path = trained_tokenizer / "spm.vocab"
    vocab_content = vocab_path.read_text(encoding="utf-8")

    assert "<pad>" in vocab_content
    assert "<unk>" in vocab_content
    assert "<s>" in vocab_content
    assert "</s>" in vocab_content


def test_spm_training_is_deterministic(tmp_path, sample_text_file):
    save_path_1 = tmp_path / "tokenizer_1"
    save_path_2 = tmp_path / "tokenizer_2"

    tokenizer_1 = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer_1.train(sample_text_file, save_path_1)

    tokenizer_2 = SentencePieceTokenizer(vocab_size=200, min_frequency=1)
    tokenizer_2.train(sample_text_file, save_path_2)

    vocab_1 = (save_path_1 / "spm.vocab").read_text(encoding="utf-8")
    vocab_2 = (save_path_2 / "spm.vocab").read_text(encoding="utf-8")

    assert vocab_1 == vocab_2

    model_1 = (save_path_1 / "spm.model").read_bytes()
    model_2 = (save_path_2 / "spm.model").read_bytes()

    assert model_1 == model_2
