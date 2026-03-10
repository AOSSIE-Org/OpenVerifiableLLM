from pathlib import Path

import sentencepiece as spm

from .base import BaseTokenizer


SPM_MODEL_FILE = "spm.model"
SPM_VOCAB_FILE = "spm.vocab"


class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tokenizer implementation.

    Supports training a BPE SentencePiece tokenizer,
    encoding text to token ids, decoding token ids back to text,
    and loading a previously trained model from disk.

    Reproducibility depends on:
    - Stable input data
    - Pinned sentencepiece library version
    - Consistent execution environment
    """

    def __init__(self, vocab_size: int, min_frequency: int):
        super().__init__(vocab_size, min_frequency)
        self._model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text_file: Path, save_path: Path):
        """
        Train SentencePiece model on text corpus and save artifacts.

        Args:
            text_file: Path to training text corpus.
            save_path: Directory to save spm.model and spm.vocab.

        Raises:
            FileNotFoundError: If text_file does not exist
                               or is not a file.
        """

        text_file = Path(text_file)
        save_path = Path(save_path)

        if not text_file.is_file():
            raise FileNotFoundError(
                f"Training file not found at {text_file}. "
                f"Please provide a valid text corpus file."
            )

        save_path.mkdir(parents=True, exist_ok=True)

        model_prefix = save_path / "spm"

        spm.SentencePieceTrainer.train(
            input=str(text_file),
            model_prefix=str(model_prefix),
            vocab_size=self.vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            character_coverage=1.0,
            model_type="bpe",
        )

        self._load_model(save_path)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list:
        """
        Encode text into list of token ids.

        Args:
            text: Input string to tokenize.

        Returns:
            List of integer token ids.

        Raises:
            RuntimeError: If tokenizer has not been trained or loaded.
        """

        self._check_loaded()
        return self._model.encode(text, out_type=int)

    def decode(self, ids: list) -> str:
        """
        Decode list of token ids back into text.

        Args:
            ids: List of integer token ids.

        Returns:
            Decoded string.

        Raises:
            RuntimeError: If tokenizer has not been trained or loaded.
        """

        self._check_loaded()
        return self._model.decode(ids)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, tokenizer_dir: Path):
        """
        Load a previously trained SentencePiece model from disk.

        Args:
            tokenizer_dir: Directory containing spm.model

        Raises:
            FileNotFoundError: If spm.model is not found.
        """

        tokenizer_dir = Path(tokenizer_dir)
        self._load_model(tokenizer_dir)

    # ------------------------------------------------------------------
    # Artifact paths
    # ------------------------------------------------------------------

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        """Return path to spm.vocab file."""
        return Path(tokenizer_dir) / SPM_VOCAB_FILE

    def get_merges_path(self, tokenizer_dir: Path):
        """
        SentencePiece does not use a merges file.
        Returns None for compatibility with BaseTokenizer interface.
        """
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, tokenizer_dir: Path):
        """
        Internal helper to load spm.model from directory.

        Raises:
            FileNotFoundError: If spm.model not found.
        """

        model_path = Path(tokenizer_dir) / SPM_MODEL_FILE

        if not model_path.is_file():
            raise FileNotFoundError(
                f"SentencePiece model not found at {model_path}. "
                f"Please train the tokenizer first."
            )

        self._model = spm.SentencePieceProcessor()
        self._model.load(str(model_path))

    def _check_loaded(self):
        """
        Check that model is loaded before encode/decode.

        Raises:
            RuntimeError: If model has not been loaded or trained.
        """

        if self._model is None:
            raise RuntimeError(
                "SentencePiece model is not loaded. "
                "Call train() or load() before encode/decode."
            )