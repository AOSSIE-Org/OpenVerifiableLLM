import bz2
import re
import defusedxml.ElementTree as ET
from pathlib import Path
import sys
from typing import Union, Optional
import hashlib
import logging
import json
import platform

logger = logging.getLogger(__name__)

MERKLE_CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB


# ---------------------------------------------------------------------
# SHA256 (Backward Compatible)
# ---------------------------------------------------------------------
def compute_sha256(
    file_path: Optional[Union[str, Path, bytes, bytearray]] = None,
    *,
    data: Optional[Union[bytes, bytearray]] = None,
) -> str:
    """
    Compute SHA256 hash of a file OR raw bytes.

    Backward compatible:
    - compute_sha256("file.txt")
    - compute_sha256(Path(...))
    - compute_sha256(b"bytes")
    - compute_sha256(data=b"bytes")
    """

    if file_path is None and data is None:
        raise ValueError("Either file_path or data must be provided.")

    sha256 = hashlib.sha256()

    # If keyword data is used
    if data is not None:
        sha256.update(data)
        return sha256.hexdigest()

    # If positional argument is bytes
    if isinstance(file_path, (bytes, bytearray)):
        sha256.update(file_path)
        return sha256.hexdigest()

    # Otherwise treat as file path
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ---------------------------------------------------------------------
# Merkle Tree Root
# ---------------------------------------------------------------------
def compute_merkle_root(
    file_path: Union[str, Path],
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES,
) -> str:
    """
    Compute Merkle root of a file using chunk-level hashing.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    leaves = []

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            leaf_hex = compute_sha256(data=chunk)
            leaves.append(bytes.fromhex(leaf_hex))

    # Empty file case
    if not leaves:
        return compute_sha256(data=b"")

    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left

            combined = left + right
            parent_hex = compute_sha256(data=combined)
            next_level.append(bytes.fromhex(parent_hex))

        leaves = next_level

    return leaves[0].hex()


# ---------------------------------------------------------------------
# Wikipedia XML Processing
# ---------------------------------------------------------------------
def extract_text_from_xml(input_path):
    """
    Process a compressed Wikipedia XML dump into cleaned plain text.
    Output saved to data/processed/wiki_clean.txt
    """

    input_path = Path(input_path)

    project_root = Path.cwd()
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "wiki_clean.txt"

    with bz2.open(input_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        with open(output_path, "w", encoding="utf-8") as out:
            for _, elem in context:
                if elem.tag.endswith("page"):
                    text_elem = elem.find(".//{*}text")

                    if text_elem is not None and text_elem.text:
                        cleaned = clean_wikitext(text_elem.text)
                        if cleaned:
                            out.write(cleaned + "\n\n")

                    elem.clear()

    logger.info("Preprocessing complete. Output saved to %s", output_path)
    generate_manifest(input_path, output_path)


# ---------------------------------------------------------------------
# Manifest Generation
# ---------------------------------------------------------------------
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
        "raw_sha256": compute_sha256(raw_path),
        "processed_sha256": compute_sha256(processed_path),
        "raw_merkle_root": compute_merkle_root(raw_path),
        "processed_merkle_root": compute_merkle_root(processed_path),
        "chunk_size_bytes": MERKLE_CHUNK_SIZE_BYTES,
        "preprocessing_version": "v1",
        "python_version": platform.python_version(),
    }

    project_root = Path.cwd()
    manifest_path = project_root / "data" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def extract_dump_date(filename: str):
    parts = filename.split("-")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    return "unknown"


def clean_wikitext(text: str) -> str:
    """
    Basic deterministic wikitext cleaning.
    """

    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openverifiablellm.utils <input_dump>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    extract_text_from_xml(sys.argv[1])
