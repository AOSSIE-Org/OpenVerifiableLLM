import json
import platform
from pathlib import Path
from .hash_utils import sha256_file

# Anchor paths to project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_dump_date(filename: str):
    parts = filename.split("-")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    return "unknown"


def generate_manifest(raw_path, processed_path):
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed file not found at {processed_path}. Run preprocessing first."
        )

    manifest = {
        "wikipedia_dump": raw_path.name,
        "dump_date": extract_dump_date(raw_path.name),
        "raw_sha256": sha256_file(str(raw_path)),
        "processed_sha256": sha256_file(str(processed_path)),
        "preprocessing_version": "v1",
        "python_version": platform.python_version()
    }

    manifest_path = _PROJECT_ROOT / "dataset_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {manifest_path}")