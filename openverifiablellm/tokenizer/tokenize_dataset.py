from pathlib import Path
import numpy as np


def tokenize_dataset(input_file, tokenizer, output_file):
    """
    Tokenize a dataset using a trained tokenizer and save tokens to a binary file.

    This implementation is streaming and memory-efficient, meaning it can handle
    very large datasets without loading everything into memory.

    Parameters
    ----------
    input_file : str or Path
        Path to the cleaned dataset text file.

    tokenizer : object
        A tokenizer instance with an `encode()` method.

    output_file : str or Path
        Path where tokenized binary output will be written.

    Returns
    -------
    int
        Total number of tokens written.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    total_tokens = 0

    # open dataset for streaming
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("wb") as fout:

        for line in fin:

            text = line.strip()

            if not text:
                continue

            encoded = tokenizer.encode(text)

            if isinstance(encoded, list):
                tokens = encoded
            else:
                # support tokenizers returning objects
                tokens = encoded.ids

            tokens_array = np.array(tokens, dtype=np.uint32)

            tokens_array.tofile(fout)

            total_tokens += len(tokens)

    return total_tokens
