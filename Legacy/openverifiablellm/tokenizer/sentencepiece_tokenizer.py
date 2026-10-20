from pathlib import Path
from typing import List, Optional

import sentencepiece as spm

from .base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tokenizer implementation.
    """

    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        super().__init__(vocab_size, min_frequency)
        self._sp: Optional[spm.SentencePieceProcessor] = None

    def train(self, text_file: Path, save_path: Path):
        text_file = Path(text_file)
        save_path = Path(save_path)
        if not text_file.is_file():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        save_path.mkdir(parents=True, exist_ok=True)
        model_prefix = save_path / "spm"

        spm.SentencePieceTrainer.train(
            input=str(text_file),
            model_prefix=str(model_prefix),
            vocab_size=self.vocab_size,
        )

        model_file = save_path / "spm.model"
        if model_file.is_file():
            self._sp = spm.SentencePieceProcessor(model_file=str(model_file))

    def load(self, tokenizer_dir: Path):
        tokenizer_dir = Path(tokenizer_dir)
        model_file = self.get_model_path(tokenizer_dir)
        if not model_file.is_file():
            raise FileNotFoundError(f"spm.model not found at {model_file}")

        self._sp = spm.SentencePieceProcessor(model_file=str(model_file))

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if self._sp is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call train() or load() first.")
        return [int(tok) for tok in self._sp.encode(text, out_type=int)]

    def decode(self, token_ids: List[int]) -> str:
        if self._sp is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call train() or load() first.")
        return self._sp.decode([int(t) for t in token_ids])

    def get_model_path(self, tokenizer_dir: Path) -> Path:
        return Path(tokenizer_dir) / "spm.model"

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        return Path(tokenizer_dir) / "spm.vocab"

    def get_merges_path(self, tokenizer_dir: Path) -> Optional[Path]:
        # SentencePiece does not use merges
        return None
