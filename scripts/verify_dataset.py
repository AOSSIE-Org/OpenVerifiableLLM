import argparse
import json
import logging
import sys
from pathlib import Path

from openverifiablellm.verify import verify_preprocessing

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Verify deterministic preprocessing and manifest integrity."
    )

    parser.add_argument(
        "input_dump",
        help="Path to raw Wikipedia dump",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to dataset_manifest.json",
    )

    parser.add_argument(
        "--previous-manifest",
        default=None,
        help="Previous manifest for chain verification",
    )

    parser.add_argument(
        "--json",
        dest="output_json",
        default=None,
        help="Optional JSON report output",
    )

    args = parser.parse_args()

    report = verify_preprocessing(
        input_dump=args.input_dump,
        manifest_path=args.manifest,
        previous_manifest_path=args.previous_manifest,
    )

    print(report.summary())

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"\nJSON report written to: {out}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
