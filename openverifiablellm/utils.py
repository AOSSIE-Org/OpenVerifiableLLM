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
MERKLE_CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB

# Merkle Tree Chunk-Level Hashing for Large Files
def compute_merkle_root(file_path: Union[str, Path], chunk_size: int = MERKLE_CHUNK_SIZE_BYTES) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    leaves = []

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            # reuse compute_sha256
            leaf_hex = compute_sha256(chunk)
            leaves.append(bytes.fromhex(leaf_hex))

    if not leaves:
        return compute_sha256(b"")

    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left

            combined = left + right
            parent_hex = compute_sha256(combined)
            next_level.append(bytes.fromhex(parent_hex))

        leaves = next_level

    return leaves[0].hex()

def generate_merkle_proof(
    file_path: Union[str, Path],
    chunk_index: int,
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES
):
    """
    Generate Merkle proof for a specific chunk index.

    Returns:
        List of tuples (sibling_hash_hex, is_left)
    """
    path = Path(file_path)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    leaves = []

    # Build leaf level
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            leaf_hex = compute_sha256(chunk)
            leaves.append(bytes.fromhex(leaf_hex))

    if not leaves:
        raise ValueError("Cannot generate proof for empty file")

    if chunk_index < 0 or chunk_index >= len(leaves):
        raise IndexError("Chunk index out of range")

    proof = []
    index = chunk_index

    while len(leaves) > 1:
        # If odd number of nodes, duplicate last
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])

        sibling_index = index ^ 1
        sibling = leaves[sibling_index]

        is_left = sibling_index < index
        proof.append((sibling.hex(), is_left))

        # Build next level
        next_level = []
        for i in range(0, len(leaves), 2):
            combined = leaves[i] + leaves[i + 1]
            parent_hex = compute_sha256(combined)
            next_level.append(bytes.fromhex(parent_hex))

        index //= 2
        leaves = next_level

    return proof

def verify_merkle_proof(
    chunk_bytes: bytes,
    proof,
    merkle_root: str
) -> bool:
    """
    Verify a Merkle proof for given chunk bytes.
    """
    try:
        current_hash = bytes.fromhex(compute_sha256(chunk_bytes))
        expected_root = bytes.fromhex(merkle_root)
    except (TypeError, ValueError):
        return False

    if not isinstance(proof, (list, tuple)):
        return False

    for step in proof:
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            return False

        sibling_hex, is_left = step

        if not isinstance(sibling_hex, str) or not isinstance(is_left, bool):
            return False

        try:
            sibling = bytes.fromhex(sibling_hex)
        except (TypeError, ValueError):
            return False

        # Ensure correct hash length
        if len(sibling) != hashlib.sha256().digest_size:
            return False

        if is_left:
            combined = sibling + current_hash
        else:
            combined = current_hash + sibling

        parent_hex = compute_sha256(combined)
        current_hash = bytes.fromhex(parent_hex)

    return current_hash == expected_root

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
        "raw_merkle_root": compute_merkle_root(raw_path, chunk_size=MERKLE_CHUNK_SIZE_BYTES),
        "processed_merkle_root": compute_merkle_root(processed_path, chunk_size=MERKLE_CHUNK_SIZE_BYTES),
        "chunk_size_bytes": MERKLE_CHUNK_SIZE_BYTES,
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

# helpers:Update compute_sha256() to support bytes input directly.
def compute_sha256(data: Union[str, Path, bytes, bytearray]) -> str:
    """
    Compute SHA256 hash of a file OR raw bytes.
        This is used for both raw and processed files to ensure integrity.
    This provides a deterministic fingerprint of the dataset,
    enabling reproducibility and verification.
    """
    sha256 = hashlib.sha256()

    # If raw bytes provided
    if isinstance(data, (bytes, bytearray)):
        sha256.update(data)
        return sha256.hexdigest()

    path = Path(data)

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
