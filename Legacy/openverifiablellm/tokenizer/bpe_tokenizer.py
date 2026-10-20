from pathlib import Path
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from .base import BaseTokenizer

SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        super().__init__(vocab_size, min_frequency)
        self._tokenizer: Optional[ByteLevelBPETokenizer] = None

    def train(self, text_file: Path, save_path: Path):
        text_file = Path(text_file)
        save_path = Path(save_path)
        if not text_file.is_file():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=[str(text_file)],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )

        tokenizer.save_model(str(save_path))
        self._tokenizer = tokenizer

    def load(self, tokenizer_dir: Path):
        tokenizer_dir = Path(tokenizer_dir)
        vocab_path = self.get_vocab_path(tokenizer_dir)
        merges_path = self.get_merges_path(tokenizer_dir)

        if not vocab_path.is_file():
            raise FileNotFoundError(f"vocab.json not found at {vocab_path}")
        if not merges_path.is_file():
            raise FileNotFoundError(f"merges.txt not found at {merges_path}")

        self._tokenizer = ByteLevelBPETokenizer.from_file(str(vocab_path), str(merges_path))

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call train() or load() first.")
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call train() or load() first.")
        return self._tokenizer.decode(token_ids)

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        return Path(tokenizer_dir) / "vocab.json"

    def get_merges_path(self, tokenizer_dir: Path) -> Path:
        return Path(tokenizer_dir) / "merges.txt"
