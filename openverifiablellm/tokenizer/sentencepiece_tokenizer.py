from pathlib import Path

import sentencepiece as spm

from .base import BaseTokenizer

SPM_MODEL_FILE = "spm.model"
SPM_VOCAB_FILE = "spm.vocab"


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int, min_frequency: int):
        super().__init__(vocab_size, min_frequency)
        self._model = None

    def train(self, text_file: Path, save_path: Path):
        text_file = Path(text_file)
        save_path = Path(save_path)

        if not text_file.is_file():
            raise FileNotFoundError(
                f"Training file not found at {text_file}. Please provide a valid text corpus file."
            )

        if self.min_frequency != 1:
            raise NotImplementedError(
                f"min_frequency={self.min_frequency} is not supported. "
                "SentencePiece does not expose a confirmed min_count option via the Python wrapper. "
                "Set min_frequency=1 to use the default behaviour, or confirm the upstream option before enabling filtering."
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

    def encode(self, text: str) -> list:
        self._check_loaded()
        return self._model.encode(text, out_type=int)

    def decode(self, ids: list) -> str:
        self._check_loaded()
        return self._model.decode(ids)

    def load(self, tokenizer_dir: Path):
        tokenizer_dir = Path(tokenizer_dir)
        self._load_model(tokenizer_dir)

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        return Path(tokenizer_dir) / SPM_VOCAB_FILE

    def get_merges_path(self, tokenizer_dir: Path) -> Path:
        return Path(tokenizer_dir) / SPM_MODEL_FILE

    def _load_model(self, tokenizer_dir: Path):
        model_path = Path(tokenizer_dir) / SPM_MODEL_FILE

        if not model_path.is_file():
            raise FileNotFoundError(
                f"SentencePiece model not found at {model_path}. Please train the tokenizer first."
            )

        self._model = spm.SentencePieceProcessor()
        self._model.load(str(model_path))

    def _check_loaded(self):
        if self._model is None:
            raise RuntimeError(
                "SentencePiece model is not loaded. Call train() or load() before encode/decode."
            )
