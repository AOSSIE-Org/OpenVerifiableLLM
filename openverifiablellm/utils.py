import bz2
import re
import defusedxml.ElementTree as ET
from pathlib import Path
import sys
from typing import Union
import hashlib
import logging
import json
import platform

logger = logging.getLogger(__name__)

# Merkle Tree Chunk-Level Hashing for Large Files
def compute_merkle_root(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Compute the Merkle Root hash of a file by splitting it into chunks.

    This allows researchers to cryptographically verify specific chunks
    or subsets of the training data without re-hashing the entire dataset.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the dataset file.
    chunk_size : int
        Size of each chunk in bytes (default: 1MB).

    Returns
    -------
    str
        The final Merkle Root hash string.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    leaves = []

    # 1. Read file in chunks and hash each chunk (raw bytes)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            leaves.append(hashlib.sha256(chunk).digest())

    # Handle empty files deterministically
    if not leaves:
        return hashlib.sha256(b"").hexdigest()

    # 2. Build the tree bottom-up
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left

            combined = left + right
            next_level.append(hashlib.sha256(combined).digest())

        leaves = next_level

    return leaves[0].hex()


# extract clean wikipage from actual wikipage
def extract_text_from_xml(input_path):
    """
    Process a compressed Wikipedia XML dump into cleaned plain text.

    Each <page> element is parsed, its revision text is extracted,
    cleaned using `clean_wikitext()`, and appended to a single
    output text file.

    The processed output is saved to:
        data/processed/wiki_clean.txt

    Parameters
    ----------
    input_path : str or Path
        Path to the compressed Wikipedia XML (.bz2) dump file.

    Output
    ------
    Creates:
        data/processed/wiki_clean.txt
    """
    input_path = Path(input_path)

    # Fixed output path
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
    generate_manifest(input_path,output_path)
    
# generate data manifest
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
        "raw_sha256": compute_sha256(str(raw_path)),
        "processed_sha256": compute_sha256(str(processed_path)),

        # ---------------- ADDED FIELDS ----------------
        "raw_merkle_root": compute_merkle_root(raw_path),
        "processed_merkle_root": compute_merkle_root(processed_path),
        "chunk_size_bytes": 1024 * 1024,
        # ---------------------------------------------------------------

        "preprocessing_version": "v1",
        "python_version": platform.python_version()
    }
    project_root = Path.cwd()
    manifest_path = project_root / "data" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)

# helpers
def compute_sha256(file_path: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of a file.

    This provides a deterministic fingerprint of the dataset,
    enabling reproducibility and verification.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the dataset file (string or Path-like).

    Returns
    -------
    str
        SHA256 hash string.
    """
    path = Path(file_path)

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()

def extract_dump_date(filename: str):
    parts = filename.split("-")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    return "unknown"

def clean_wikitext(text: str) -> str:
    """
    Basic deterministic wikitext cleaning.

    Note:
    This uses simple regex-based rules for speed and consistency.
    It does NOT fully parse MediaWiki syntax.

    Limitations:
    - Deeply nested templates may not be fully removed.
    - Some complex <ref /> cases may not be perfectly handled.
    - This is not a complete MediaWiki parser.

    These limitations are acceptable for lightweight, deterministic preprocessing.
    """
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openverifiablellm.utils <input_dump>")
        sys.exit(1)
        
    logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
    )
    extract_text_from_xml(sys.argv[1])
