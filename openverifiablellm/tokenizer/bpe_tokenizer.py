from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

from .base import BaseTokenizer


SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]


class BPETokenizer(BaseTokenizer):

    def train(self, text_file: Path, save_path: Path):

        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=[str(text_file)],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )

        tokenizer.save_model(str(save_path))

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        return tokenizer_dir / "vocab.json"

    def get_merges_path(self, tokenizer_dir: Path) -> Path:
        return tokenizer_dir / "merges.txt"
