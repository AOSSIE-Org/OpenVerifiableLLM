from .bpe_tokenizer import BPETokenizer
from .sentencepiece_tokenizer import SentencePieceTokenizer


def create_tokenizer(tokenizer_type, vocab_size, min_frequency):
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == "bpe":
        return BPETokenizer(vocab_size, min_frequency)

    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(vocab_size, min_frequency)

    raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")
