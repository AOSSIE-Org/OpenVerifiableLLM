from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

from .base import BaseTokenizer


SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]


class BPETokenizer(BaseTokenizer):
    """
    Byte-level BPE tokenizer implementation.

    Wraps HuggingFace's ByteLevelBPETokenizer and implements
    the full BaseTokenizer interface including train, encode,
    decode, and load.
    """

    def __init__(self, vocab_size: int, min_frequency: int):
        super().__init__(vocab_size, min_frequency)
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text_file: Path, save_path: Path):
        """
        Train BPE tokenizer on text corpus and save artifacts.

        Args:
            text_file: Path to training text corpus.
            save_path: Directory to save vocab.json and merges.txt.

        Raises:
            FileNotFoundError: If text_file does not exist or is not a file.
        """

        text_file = Path(text_file)
        save_path = Path(save_path)

        if not text_file.is_file():
            raise FileNotFoundError(
                f"Training file not found at {text_file}. "
                f"Please provide a valid text corpus file."
            )

        save_path.mkdir(parents=True, exist_ok=True)

        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=[str(text_file)],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )

        # Must create directory BEFORE save_model() is called
        save_path.mkdir(parents=True, exist_ok=True)

        tokenizer.save_model(str(save_path))

        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list:
        """
        Encode text into a list of token ids.

        Args:
            text: Input string to tokenize.

        Returns:
            List of integer token ids.

        Raises:
            RuntimeError: If tokenizer has not been trained or loaded.
        """

        self._check_loaded()
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        """
        Decode a list of token ids back into text.

        Args:
            ids: List of integer token ids.

        Returns:
            Decoded string.

        Raises:
            RuntimeError: If tokenizer has not been trained or loaded.
        """

        self._check_loaded()
        return self._tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, tokenizer_dir: Path):
        """
        Load a previously trained BPE tokenizer from disk.

        Args:
            tokenizer_dir: Directory containing vocab.json and merges.txt.

        Raises:
            FileNotFoundError: If vocab.json or merges.txt are not found.
        """

        tokenizer_dir = Path(tokenizer_dir)

        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"

        if not vocab_path.is_file():
            raise FileNotFoundError(
                f"vocab.json not found at {vocab_path}. "
                f"Please train the tokenizer first."
            )

        if not merges_path.is_file():
            raise FileNotFoundError(
                f"merges.txt not found at {merges_path}. "
                f"Please train the tokenizer first."
            )

        self._tokenizer = ByteLevelBPETokenizer(
            vocab=str(vocab_path),
            merges=str(merges_path),
        )

    # ------------------------------------------------------------------
    # Artifact paths
    # ------------------------------------------------------------------

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        """Return path to vocab.json file."""
        return Path(tokenizer_dir) / "vocab.json"

    def get_merges_path(self, tokenizer_dir: Path) -> Path:
        """Return path to merges.txt file."""
        return Path(tokenizer_dir) / "merges.txt"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_loaded(self):
        """
        Check that tokenizer is loaded before encode/decode.

        Raises:
            RuntimeError: If tokenizer has not been trained or loaded.
        """

        if self._tokenizer is None:
            raise RuntimeError(
                "BPE tokenizer is not loaded. "
                "Call train() or load() before encode/decode."
            )