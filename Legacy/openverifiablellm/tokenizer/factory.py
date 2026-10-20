from pathlib import Path
from typing import Optional, Union

from .base import BaseTokenizer
from .bpe_tokenizer import BPETokenizer
from .sentencepiece_tokenizer import SentencePieceTokenizer


def create_tokenizer(
    tokenizer_type: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> BaseTokenizer:
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == "bpe":
        return BPETokenizer(vocab_size, min_frequency)

    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(vocab_size, min_frequency)

    raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")


def load_tokenizer(
    tokenizer_dir: Union[str, Path],
    tokenizer_type: Optional[str] = None,
) -> BaseTokenizer:
    """
    Load a trained tokenizer from directory.

    If tokenizer_type is not provided, automatically detects whether
    SentencePiece (spm.model) or BPE (vocab.json, merges.txt) artifacts are present.
    """
    tokenizer_dir = Path(tokenizer_dir)
    if not tokenizer_dir.is_dir():
        raise NotADirectoryError(f"Tokenizer directory not found: {tokenizer_dir}")

    t_type = tokenizer_type.lower() if tokenizer_type else None

    has_spm = (tokenizer_dir / "spm.model").is_file()
    has_bpe = (tokenizer_dir / "vocab.json").is_file()

    if t_type == "sentencepiece" or (t_type is None and has_spm and not has_bpe):
        tok = SentencePieceTokenizer()
        tok.load(tokenizer_dir)
        return tok

    if t_type == "bpe" or (t_type is None and has_bpe):
        tok = BPETokenizer()
        tok.load(tokenizer_dir)
        return tok

    if t_type is not None:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    raise FileNotFoundError(
        f"Could not identify tokenizer artifacts in {tokenizer_dir}. "
        "Expected spm.model for SentencePiece or vocab.json/merges.txt for BPE."
    )
