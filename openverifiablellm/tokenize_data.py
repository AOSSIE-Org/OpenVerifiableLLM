import argparse
import logging
import time
from pathlib import Path

import numpy as np
from tokenizers.implementations import ByteLevelBPETokenizer

from openverifiablellm.utils import compute_sha256

logger = logging.getLogger(__name__)


def tokenize_dataset(
    text_path: Path, tokenizer_path: Path, output_path: Path, chunk_size: int = 1024 * 1024 * 10
) -> str:
    """
    Tokenizes a large text dataset into a deterministic uint16 binary file.
    Reads chunk by chunk and streams encoded tokens.
    Uses little-endian uint16 to ensure the output binary hash is identical
    across different CPU architectures (ARM vs x86).
    """
    logger.info("Loading tokenizer from %s", tokenizer_path)

    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_path}")

    tokenizer = ByteLevelBPETokenizer(vocab=str(vocab_path), merges=str(merges_path))

    logger.info("Tokenizing %s to %s", text_path, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_tokens = 0
    total_chunks = 0

    with open(text_path, encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        remainder = ""
        while True:
            text_chunk = f_in.read(chunk_size)
            if not text_chunk:
                if remainder:
                    encoded = tokenizer.encode(remainder)
                    if encoded.ids and max(encoded.ids) > 65535:
                        raise ValueError(
                            f"Token ID {max(encoded.ids)} exceeds uint16 bound of 65535"
                        )
                    ids = np.array(encoded.ids, dtype="<u2")
                    f_out.write(ids.tobytes())
                    total_tokens += len(ids)
                break

            text_chunk = remainder + text_chunk

            last_space_idx = text_chunk.rfind(" ")
            last_newline_idx = text_chunk.rfind("\n")
            split_idx = max(last_space_idx, last_newline_idx)

            if split_idx != -1:
                complete_portion = text_chunk[:split_idx]
                remainder = text_chunk[split_idx:]
            else:
                complete_portion = ""
                remainder = text_chunk

            if not complete_portion:
                continue

            encoded = tokenizer.encode(complete_portion)

            if encoded.ids and max(encoded.ids) > 65535:
                raise ValueError(f"Token ID {max(encoded.ids)} exceeds uint16 bound of 65535")

            # Strict little-endian uint16 ('<u2') for verifiable determinism
            ids = np.array(encoded.ids, dtype="<u2")

            f_out.write(ids.tobytes())

            total_tokens += len(ids)
            total_chunks += 1
            if total_chunks % 10 == 0:
                logger.info("Processed %d chunks, %d total tokens", total_chunks, total_tokens)

    elapsed = time.time() - start_time
    logger.info("Tokenization complete! Total tokens: %d, Time: %.2fs", total_tokens, elapsed)

    logger.info("Computing SHA256 of output binary...")
    output_hash = compute_sha256(file_path=output_path)
    logger.info("Binary Hash: %s", output_hash)

    return output_hash


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize cleaned dataset into binary format deterministically"
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="Path to cleaned text file (e.g. wiki_clean.txt)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to trained tokenizer directory"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output binary file (.bin)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    tokenize_dataset(Path(args.text_path), Path(args.tokenizer_path), Path(args.output_path))


if __name__ == "__main__":
    main()
