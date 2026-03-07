import json
import logging
from pathlib import Path
from typing import Union

from openverifiablellm.utils import compute_sha256
from .factory import create_tokenizer

logger = logging.getLogger(__name__)

TOKENIZER_VOCAB_SIZE = 32000
TOKENIZER_MIN_FREQUENCY = 2


def train_tokenizer(
    text_file: Union[str, Path],
    save_path: Union[str, Path] = "data/tokenizer",
    tokenizer_type: str = "bpe",
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    min_frequency: int = TOKENIZER_MIN_FREQUENCY,
) -> Path:
    """
    Train a tokenizer on preprocessed text.

    Currently supports:
    - BPE
    - SentencePiece

    Reproducibility depends on:
    - Stable input data
    - Stable file ordering
    - Pinned tokenizer library versions
    - Consistent execution environment
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    if min_frequency <= 0:
        raise ValueError("min_frequency must be > 0")

    text_file = Path(text_file)
    save_path = Path(save_path)

    if not text_file.is_file():
        raise FileNotFoundError(
            f"Text file not found at {text_file}. Run preprocessing first."
        )

    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Training %s tokenizer on %s", tokenizer_type, text_file)

    tokenizer = create_tokenizer(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    tokenizer.train(text_file, save_path)

    logger.info("Tokenizer saved to %s", save_path)

    return save_path


def hash_tokenizer_config(tokenizer_path: Union[str, Path]) -> dict:
    """
    Compute SHA256 hashes of tokenizer configuration files.
    """

    tokenizer_path = Path(tokenizer_path)

    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"

    if not vocab_path.is_file():
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")

    if not merges_path.is_file():
        raise FileNotFoundError(f"merges.txt not found at {merges_path}")

    vocab_bytes = vocab_path.read_bytes()
    vocab_hash = compute_sha256(data=vocab_bytes)
    actual_vocab_size = len(json.loads(vocab_bytes.decode("utf-8")))

    merges_hash = compute_sha256(file_path=merges_path)

    logger.info("Tokenizer config hashed successfully")

    return {
        "tokenizer_vocab_hash": vocab_hash,
        "tokenizer_merges_hash": merges_hash,
        "tokenizer_vocab_size": actual_vocab_size,
    }
