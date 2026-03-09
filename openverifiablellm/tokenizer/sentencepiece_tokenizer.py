from pathlib import Path
import sentencepiece as spm

from .base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tokenizer implementation.
    """

    def train(self, text_file: Path, save_path: Path):

        model_prefix = save_path / "spm"

        spm.SentencePieceTrainer.train(
            input=str(text_file),
            model_prefix=str(model_prefix),
            vocab_size=self.vocab_size,
        )

    def get_vocab_path(self, tokenizer_dir: Path):
        return tokenizer_dir / "spm.vocab"

    def get_merges_path(self, tokenizer_dir: Path):
        # SentencePiece does not use merges
        return None
