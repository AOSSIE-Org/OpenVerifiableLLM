import json
import sys
import platform
from pathlib import Path
from hash_utils import sha256_file


def extract_dump_date(filename: str):
    parts = filename.split("-")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    return "unknown"


def generate_manifest(raw_path, processed_path):
    raw_path = Path(raw_path)

    # Automatically infer processed file path
    processed_path = Path("data/processed/wiki_clean.txt")

    if not processed_path.exists():
        print("Error: Processed file not found. Run preprocessing first.")
        sys.exit(1)

    manifest = {
        "wikipedia_dump": raw_path.name,
        "dump_date": extract_dump_date(raw_path.name),
        "raw_sha256": sha256_file(str(raw_path)),
        "processed_sha256": sha256_file(str(processed_path)),
        "preprocessing_version": "v1",
        "python_version": platform.python_version()
    }

    with open("dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Manifest generated successfully.")
