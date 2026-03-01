import json
import logging
from pathlib import Path
from typing import Union

from tokenizers import ByteLevelBPETokenizer

from openverifiablellm.utils import compute_sha256

logger = logging.getLogger(__name__)

TOKENIZER_VOCAB_SIZE = 32000
TOKENIZER_MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]


def train_tokenizer(
    text_file: Union[str, Path],
    save_path: Union[str, Path] = "data/tokenizer",
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    min_frequency: int = TOKENIZER_MIN_FREQUENCY,
) -> Path:
    """
    Train a deterministic Byte-Level BPE tokenizer on preprocessed Wikipedia text.

    Uses fixed vocabulary size and configuration to ensure determinism —
    same input always produces identical vocab.json and merges.txt files.

    Parameters
    ----------
    text_file : str or Path
        Path to preprocessed Wikipedia text file (output of extract_text_from_xml)
    save_path : str or Path
        Directory to save tokenizer files (vocab.json and merges.txt)
    vocab_size : int
        Fixed vocabulary size for determinism (default 32000)
    min_frequency : int
        Minimum frequency for a token to be included (default 2)

    Returns
    -------
    Path
        Path to directory containing saved tokenizer files
    """
    text_file = Path(text_file)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not text_file.exists():
        raise FileNotFoundError(
            f"Text file not found at {text_file}. Run preprocessing first."
        )

    logger.info("Training BPE tokenizer on %s", text_file)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(text_file)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.save_model(str(save_path))
    logger.info("Tokenizer saved to %s", save_path)

    return save_path


def hash_tokenizer_config(tokenizer_path: Union[str, Path]) -> dict:
    """
    Compute SHA256 hashes of tokenizer configuration files.

    Hashes vocab.json and merges.txt to create a cryptographic
    fingerprint of the tokenizer — making the tokenizer itself verifiable.
    The vocab size is derived directly from the vocab file to ensure
    the reported size matches the actual tokenizer state.

    Parameters
    ----------
    tokenizer_path : str or Path
        Directory containing vocab.json and merges.txt

    Returns
    -------
    dict
        Dictionary containing hashes of tokenizer config files
    """
    tokenizer_path = Path(tokenizer_path)
    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"

    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")
    if not merges_path.exists():
        raise FileNotFoundError(f"merges.txt not found at {merges_path}")

    vocab_hash = compute_sha256(file_path=vocab_path)
    merges_hash = compute_sha256(file_path=merges_path)

    # Derive vocab size directly from vocab file for accuracy
    actual_vocab_size = len(json.loads(vocab_path.read_text(encoding="utf-8")))

    logger.info("Tokenizer config hashed successfully")

    return {
        "tokenizer_vocab_hash": vocab_hash,
        "tokenizer_merges_hash": merges_hash,
        "tokenizer_vocab_size": actual_vocab_size,
    }
