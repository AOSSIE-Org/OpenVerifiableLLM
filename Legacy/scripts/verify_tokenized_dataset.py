import argparse
import logging
from pathlib import Path
import sys

from openverifiablellm.tokenizer import verify_tokenized_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Verify tokenized dataset binary artifact against its cryptographic manifest."
    )
    parser.add_argument(
        "tokenized_file",
        help="Path to tokenized binary dataset (.bin)",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to tokenized dataset manifest JSON",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional path to tokenizer directory to verify configuration hashes",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional path to original preprocessed text file to verify input hash",
    )
    parser.add_argument(
        "--previous-manifest",
        default=None,
        help="Optional path to previous manifest to verify cryptographic chain link",
    )

    args = parser.parse_args()

    try:
        report = verify_tokenized_dataset(
            tokenized_file=Path(args.tokenized_file),
            manifest_path=Path(args.manifest),
            tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
            input_file=Path(args.input_file) if args.input_file else None,
            previous_manifest_path=Path(args.previous_manifest) if args.previous_manifest else None,
        )

        print("\n" + report.summary() + "\n")

        if not report.all_passed:
            sys.exit(1)

    except Exception as e:
        logger.error("Verification encountered an unhandled error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
