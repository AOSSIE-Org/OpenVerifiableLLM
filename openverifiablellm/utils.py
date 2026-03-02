import bz2
import re
import defusedxml.ElementTree as ET
from pathlib import Path
import sys
import hashlib
import logging
import json
import platform
from typing import Union, Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)
MERKLE_CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB

# Precompiled regular expressions for wikitext cleaning
RE_TEMPLATE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
RE_REF = re.compile(r"<ref.*?>.*?</ref>", re.DOTALL)
RE_HTML_TAG = re.compile(r"<.*?>")
RE_LINK_PIPE = re.compile(r"\[\[.*?\|(.*?)\]\]")
RE_LINK = re.compile(r"\[\[(.*?)\]\]")
RE_WHITESPACE = re.compile(r"\s+")

# Merkle Tree Chunk-Level Hashing for Large Files
def compute_merkle_root(file_path: Union[str, Path], chunk_size: int = MERKLE_CHUNK_SIZE_BYTES) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    leaves = []

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            # reuse compute_sha256
            leaf_hex = compute_sha256(data=chunk)
            leaves.append(bytes.fromhex(leaf_hex))

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
            leaf_hex = compute_sha256(data=chunk)
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
            parent_hex = compute_sha256(data=combined)
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
        current_hash = bytes.fromhex(compute_sha256(data=chunk_bytes))
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

        current_hash = bytes.fromhex(compute_sha256(data=combined))

    return current_hash == expected_root


def compute_sha256(file_path: Union[str, Path, None] = None, data: Union[bytes, None] = None) -> str:
    """
    Compute SHA-256 hash of a file or raw bytes.
    """
    sha256 = hashlib.sha256()

    if data is not None:
        sha256.update(data)
        return sha256.hexdigest()

    if file_path is not None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    raise ValueError("Either file_path or data must be provided")


def clean_wikitext(text: str) -> str:
    """
    Clean wikitext markup from a string.
    """
    text = RE_TEMPLATE.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_HTML_TAG.sub("", text)
    text = RE_LINK_PIPE.sub(r"\1", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_WHITESPACE.sub(" ", text)
    return text.strip()


def extract_dump_date(filename: str) -> str:
    """
    Extract dump date from a Wikipedia dump filename.
    Expected format: <prefix>-YYYYMMDD-<suffix>
    Returns date as 'YYYY-MM-DD' or 'unknown'.
    """
    match = re.search(r"-(\d{8})-", filename)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return "unknown"


def extract_text_from_xml(input_path: Union[str, Path]) -> None:
    """
    Extract and clean text from a Wikipedia XML dump file.
    Supports both .bz2 compressed and plain .xml files.
    Output is written to data/processed/wiki_clean.txt.
    """
    input_path = Path(input_path)
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wiki_clean.txt"

    dump_date = extract_dump_date(input_path.name)

    suffix = input_path.suffix.lower()
    if suffix == ".bz2":
        file_handle = bz2.open(input_path, "rb")
    elif suffix == ".xml":
        file_handle = open(input_path, "rb")
    else:
        raise ValueError("input_path must have .xml or .bz2 extension")

    with file_handle as f:
        context = ET.iterparse(f, events=("end",))
        with open(output_file, "w", encoding="utf-8") as out:
            for event, elem in context:
                if elem.tag.endswith("text") and elem.text:
                    cleaned = clean_wikitext(elem.text)
                    if cleaned:
                        out.write(cleaned + "\n")
                elem.clear()

    logger.info(f"Extracted text from {input_path} (dump date: {dump_date}) to {output_file}")


def generate_manifest(
    raw_path: Union[str, Path],
    processed_path: Union[str, Path],
    output_dir: Union[str, Path, None] = None
) -> Dict[str, Any]:
    """
    Generate a dataset manifest with file metadata and hashes.
    """
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    if output_dir is None:
        output_dir = Path("data")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "dataset_manifest.json"

    manifest = {
        "raw": {
            "path": str(raw_path),
            "sha256": compute_sha256(raw_path) if raw_path.exists() else None,
            "merkle_root": compute_merkle_root(raw_path) if raw_path.exists() else None,
        },
        "processed": {
            "path": str(processed_path),
            "sha256": compute_sha256(processed_path),
            "merkle_root": compute_merkle_root(processed_path),
        },
        "platform": platform.platform(),
        "python_version": sys.version,
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {manifest_file}")
    return manifest
