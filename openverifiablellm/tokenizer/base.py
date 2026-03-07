from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizer implementations.
    """

    def __init__(self, vocab_size: int, min_frequency: int):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        if min_frequency <= 0:
            raise ValueError("min_frequency must be > 0")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    @abstractmethod
    def train(self, text_file: Path, save_path: Path):
        """Train tokenizer and save model."""
        pass

    @abstractmethod
    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        pass

    @abstractmethod
    def get_merges_path(self, tokenizer_dir: Path):
        pass
