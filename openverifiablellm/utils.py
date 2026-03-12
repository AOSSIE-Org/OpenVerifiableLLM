import argparse
import bz2
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import defusedxml.ElementTree as ET

from openverifiablellm.environment import generate_environment_fingerprint
from openverifiablellm.manifest_chain import get_parent_manifest_hash

logger = logging.getLogger(__name__)
MERKLE_CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB

# Precompiled regular expressions for wikitext cleaning
RE_TEMPLATE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
RE_REF = re.compile(r"<ref.*?>.*?</ref>", re.DOTALL)
RE_HTML_TAG = re.compile(r"<.*?>")
RE_LINK_PIPE = re.compile(r"\[\[.*?\|(.*?)\]\]")
RE_LINK = re.compile(r"\[\[(.*?)\]\]")
RE_WHITESPACE = re.compile(r"\s+")


# helpers: New helper to compute SHA256 and return raw bytes directly
def compute_sha256_bytes(
    *,
    data: Optional[Union[bytes, bytearray]] = None,
    file_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """
    Compute SHA256 hash of a file OR raw bytes, returning raw bytes.
    This avoids the overhead of converting to a hex string and back.
    """
    if (data is None) == (file_path is None):
        raise ValueError("Exactly one of 'data' or 'file_path' must be provided.")

    sha256 = hashlib.sha256()

    if data is not None:
        sha256.update(data)
        return sha256.digest()

    path = Path(file_path)
    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.digest()


# Merkle Tree Chunk-Level Hashing for Large Files
def compute_merkle_root(
    file_path: Union[str, Path], chunk_size: int = MERKLE_CHUNK_SIZE_BYTES
) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    leaves = []

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            # use new helper to directly get bytes
            leaf_bytes = compute_sha256_bytes(data=chunk)
            leaves.append(leaf_bytes)

    if not leaves:
        return compute_sha256(data=b"")

    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left

            combined = left + right
            parent_bytes = compute_sha256_bytes(data=combined)
            next_level.append(parent_bytes)

        leaves = next_level

    return leaves[0].hex()


def generate_merkle_proof(
    file_path: Union[str, Path], chunk_index: int, chunk_size: int = MERKLE_CHUNK_SIZE_BYTES
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
            leaf_bytes = compute_sha256_bytes(data=chunk)
            leaves.append(leaf_bytes)

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
            parent_bytes = compute_sha256_bytes(data=combined)
            next_level.append(parent_bytes)

        index //= 2
        leaves = next_level

    return proof


def verify_merkle_proof(chunk_bytes: bytes, proof, merkle_root: str) -> bool:
    """
    Verify a Merkle proof for given chunk bytes.
    """
    try:
        current_hash = compute_sha256_bytes(data=chunk_bytes)
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

        current_hash = compute_sha256_bytes(data=combined)

    return current_hash == expected_root


# extract clean wikipage from actual wikipage
def extract_text_from_xml(input_path, *, write_manifest: bool = False):
    """
    Process a Wikipedia XML dump (compressed or uncompressed) into cleaned plain text.

    Each <page> element is parsed, its revision text is extracted,
    cleaned using `clean_wikitext()`, and appended to a single
    output text file.

    The processed output is saved to:
        data/processed/wiki_clean.txt

    Parameters
    ----------
    input_path : str or Path
        Path to the Wikipedia XML dump file.

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

    # Auto-detect file type using magic bytes separation
    with open(input_path, "rb") as test_f:
        is_bz2 = test_f.read(3) == b"BZh"

    open_func = bz2.open if is_bz2 else open

    with open_func(input_path, "rb") as f:
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
    if write_manifest:
        generate_manifest(input_path, output_path)


# generate data manifest
def generate_manifest(raw_path, processed_path):
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed file not found at {processed_path}. Run preprocessing first."
        )

    project_root = Path.cwd()
    manifest_path = project_root / "data" / "dataset_manifest.json"

    # ===== NEW: Compute parent_manifest_hash before creating new manifest =====
    parent_manifest_hash = get_parent_manifest_hash(manifest_path)
    # ========================================================================

    manifest = {
        "wikipedia_dump": raw_path.name,
        "dump_date": extract_dump_date(raw_path.name),
        "raw_sha256": compute_sha256(file_path=raw_path),
        "processed_sha256": compute_sha256(file_path=processed_path),
        # ---------------- ADDED FIELDS ----------------
        "raw_merkle_root": compute_merkle_root(raw_path, chunk_size=MERKLE_CHUNK_SIZE_BYTES),
        "processed_merkle_root": compute_merkle_root(
            processed_path, chunk_size=MERKLE_CHUNK_SIZE_BYTES
        ),
        "chunk_size_bytes": MERKLE_CHUNK_SIZE_BYTES,
        # ---------------------------------------------------------------

        #  Add parent_manifest_hash to link to previous manifest
        "parent_manifest_hash": parent_manifest_hash,
        "preprocessing_version": "v1",
        "python_version": platform.python_version(),
    }
    
    env_data = generate_environment_fingerprint()
    manifest.update(
        {"environment": env_data["environment"], "environment_hash": env_data["environment_hash"]}
    )
    project_root = Path.cwd()
    manifest_path = project_root / "data" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)
    logger.info("Manifest parent hash: %s", parent_manifest_hash if parent_manifest_hash else "(first run)")



def export_merkle_proof(
    proof: List[Tuple[str, bool]], chunk_index: int, chunk_size: int, output_path: Union[str, Path]
) -> None:
    """
    Export Merkle proof to a JSON file for portable verification.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if not isinstance(proof, list):
        raise ValueError("proof must be a list")

    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")

    data = {
        "chunk_index": chunk_index,
        "chunk_size": chunk_size,
        "proof": proof,
    }

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_merkle_proof(proof_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load Merkle proof from a JSON file.
    """
    proof_path = Path(proof_path)

    with proof_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_merkle_proof_from_file(
    proof_file_path: Union[str, Path], chunk_data: bytes, expected_root: str
) -> bool:
    proof_file_path = Path(proof_file_path)

    if not proof_file_path.exists():
        raise FileNotFoundError(f"Proof file not found: {proof_file_path}")

    with proof_file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Malformed proof file: expected JSON object")

    required_keys = {"chunk_index", "chunk_size", "proof"}
    if not required_keys.issubset(data.keys()):
        raise ValueError("Malformed proof file: missing required keys")

    proof = data["proof"]

    if not isinstance(proof, list):
        raise ValueError("Malformed proof: proof must be a list")

    return verify_merkle_proof(chunk_data, proof, expected_root)


# helpers:Update compute_sha256() to support bytes input directly.
def compute_sha256(
    *,
    data: Optional[Union[bytes, bytearray]] = None,
    file_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Compute SHA256 hash of a file OR raw bytes.

    This is used for both raw and processed files to ensure integrity.
    This provides a deterministic fingerprint of the dataset,
    enabling reproducibility and verification.

    Exactly one of `data` or `file_path` must be provided.
    """

    if (data is None) == (file_path is None):
        raise ValueError("Exactly one of 'data' or 'file_path' must be provided.")

    sha256 = hashlib.sha256()

    if data is not None:
        sha256.update(data)
        return sha256.hexdigest()

    path = Path(file_path)
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
    text = RE_TEMPLATE.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_HTML_TAG.sub("", text)
    text = RE_LINK_PIPE.sub(r"\1", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_WHITESPACE.sub(" ", text)
    return text.strip()


def run_benchmark(file_path: str, chunk_size: int = 1024 * 1024):
    logger.info("--- Starting Benchmark ---")

    if not os.path.exists(file_path):
        logger.error(f"Error: File not found at {file_path}")
        sys.exit(1)

    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    logger.info(f"Benchmarking file: {file_path}")
    logger.info(f"File size: {size_mb:.2f} MB")

    try:
        tracemalloc.start()

        # Benchmark compute_merkle_root
        start_time = time.perf_counter()
        root_hex = compute_merkle_root(file_path, chunk_size=chunk_size)
        end_time = time.perf_counter()

        _current_mem, peak_mem = tracemalloc.get_traced_memory()

        root_time = end_time - start_time
        mins, secs = divmod(root_time, 60)
        logger.info(f"compute_merkle_root ({size_mb:.2f} MB file): {int(mins)}m {secs:.3f}s")
        logger.info(f"Peak Memory Usage: {peak_mem / 10**6:.3f} MB")
        logger.info(f"Merkle Root: {root_hex}")

        tracemalloc.reset_peak()

        # Benchmark generate_merkle_proof
        start_time = time.perf_counter()

        file_size_bytes = os.path.getsize(file_path)
        if file_size_bytes == 0:
            logger.info("Skipping proof benchmark for empty file")
        else:
            chunk_count = (file_size_bytes + chunk_size - 1) // chunk_size
            chunk_index = min(10, chunk_count - 1)

            _ = generate_merkle_proof(file_path, chunk_index=chunk_index, chunk_size=chunk_size)
            end_time = time.perf_counter()

            _, peak_mem_proof = tracemalloc.get_traced_memory()

            proof_time = end_time - start_time
            pmins, psecs = divmod(proof_time, 60)
            logger.info(
                f"generate_merkle_proof ({size_mb:.2f} MB file, chunk {chunk_index}): {int(pmins)}m {psecs:.3f}s"
            )
            logger.info(f"Peak Memory Usage for proof: {peak_mem_proof / 10**6:.3f} MB")

        logger.info("--- Benchmark Complete ---")
        tracemalloc.stop()

    except Exception:
        logger.exception("An error occurred during benchmarking")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVerifiableLLM Preprocessing")
    parser.add_argument("input_dump", help="Path to the Wikipedia XML dump file")
    parser.add_argument(
        "--BENCHMARK_MODE",
        type=str,
        choices=["TRUE", "FALSE"],
        default="FALSE",
        help="Run in benchmark mode",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=MERKLE_CHUNK_SIZE_BYTES,
        help="Chunk size in bytes for Merkle hashing",
    )
    parser.add_argument("--no-manifest", action="store_true", help="Skip manifest generation")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.BENCHMARK_MODE == "TRUE":
        run_benchmark(args.input_dump, args.chunk_size)
    else:
        extract_text_from_xml(args.input_dump, write_manifest=not args.no_manifest)
