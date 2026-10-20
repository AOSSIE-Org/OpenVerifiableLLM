import argparse
import json
import logging
from pathlib import Path
import sys

from openverifiablellm.tokenizer import tokenize_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Deterministically tokenize preprocessed dataset text into binary format with Merkle verification."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to preprocessed input text file (e.g. data/processed/wiki_clean.txt)",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to trained tokenizer directory (containing spm.model or vocab.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where tokenized binary output will be written (.bin)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to output tokenized manifest JSON",
    )
    parser.add_argument(
        "--previous-manifest",
        default=None,
        help="Optional path to previous preprocessing manifest for cryptographic chain linkage",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1048576,
        help="Merkle chunk size in bytes (default: 1048576 = 1MB)",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint16", "uint32"],
        default="uint32",
        help="Data type for token IDs (default: uint32)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing manifest file to disk",
    )

    args = parser.parse_args()

    try:
        manifest = tokenize_dataset(
            input_file=Path(args.input),
            tokenizer=Path(args.tokenizer),
            output_file=Path(args.output),
            manifest_path=Path(args.manifest) if args.manifest else None,
            previous_manifest_path=Path(args.previous_manifest) if args.previous_manifest else None,
            chunk_size_bytes=args.chunk_size,
            dtype=args.dtype,
            write_manifest=not args.no_manifest,
        )

        print("\n" + "=" * 60)
        print("TOKENIZATION COMPLETE & VERIFIED")
        print("=" * 60)
        print(f"Input file           : {manifest['input_dataset_path']}")
        print(f"Output binary file   : {manifest['tokenized_dataset_path']}")
        print(f"Total tokens         : {manifest['total_tokens']:,}")
        print(f"Total bytes          : {manifest['total_bytes']:,}")
        print(f"Token file SHA-256   : {manifest['tokenized_dataset_sha256']}")
        print(f"Merkle root          : {manifest['merkle_root']}")
        if manifest.get("parent_manifest_hash"):
            print(f"Parent manifest hash : {manifest['parent_manifest_hash']}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error("Tokenization failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
