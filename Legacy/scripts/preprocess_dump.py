import argparse
import logging
from pathlib import Path

from openverifiablellm.utils import extract_text_from_xml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Wikipedia dump and generate dataset manifest."
    )

    parser.add_argument(
        "input_dump",
        help="Path to the Wikipedia XML dump (.xml or .bz2)",
    )

    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest generation",
    )

    args = parser.parse_args()

    extract_text_from_xml(
        Path(args.input_dump),
        write_manifest=not args.no_manifest,
    )


if __name__ == "__main__":
    main()
