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
from openverifiablellm.environment import generate_environment_fingerprint


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

        parent_hex = compute_sha256(data=combined)
        current_hash = bytes.fromhex(parent_hex)

    return current_hash == expected_root

# extract clean wikipage from actual wikipage
def extract_text_from_xml(input_path):
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
    generate_manifest(input_path,output_path)

    # Automatically generate ZKP proofs for the processed text chunks
    proofs_dir = output_dir / "proofs"
    logger.info("Generating ZKP-compatible Merkle proofs for verification pipelines...")
    num_proofs = export_all_merkle_proofs(output_path, proofs_dir, MERKLE_CHUNK_SIZE_BYTES)
    logger.info("Successfully exported %d ZKP proofs to %s", num_proofs, proofs_dir)

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
        "raw_sha256": compute_sha256(file_path=raw_path),
        "processed_sha256": compute_sha256(file_path=processed_path),

        # ---------------- ADDED FIELDS ----------------
        "raw_merkle_root": compute_merkle_root(raw_path, chunk_size=MERKLE_CHUNK_SIZE_BYTES),
        "processed_merkle_root": compute_merkle_root(processed_path, chunk_size=MERKLE_CHUNK_SIZE_BYTES),
        "chunk_size_bytes": MERKLE_CHUNK_SIZE_BYTES,
        # ---------------------------------------------------------------

        "preprocessing_version": "v1",
        "python_version": platform.python_version()
    }
    env_data = generate_environment_fingerprint()
    manifest.update({
        "environment": env_data["environment"],
        "environment_hash": env_data["environment_hash"]
    })
    project_root = Path.cwd()
    manifest_path = project_root / "data" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)

def export_merkle_proof(
    proof: List[Tuple[str, bool]],
    chunk_index: int,
    chunk_size: int,
    merkle_root: str,
    output_path: Union[str, Path]
) -> None:
    """
    Export Merkle proof to a JSON file for portable verification.
    Structured into public_inputs and witness for Zero-Knowledge Proof (ZKP) compatibility.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if not isinstance(proof, list):
        raise ValueError("proof must be a list")

    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")

    data = {
        "public_inputs": {
            "merkle_root": merkle_root,
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
        },
        "witness": {
            "sibling_hashes": proof,
        }
    }

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_merkle_proof(
    proof_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load Merkle proof from a JSON file.
    """
    proof_path = Path(proof_path)

    with proof_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def export_all_merkle_proofs(
    file_path: Union[str, Path],
    output_dir: Union[str, Path],
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES
) -> int:
    """
    Efficiently generate and export Merkle proofs for all chunks of a file.
    Saves them as individual JSON files in the output directory.
    Returns the number of proofs generated.
    """
    path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    leaves = []
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            leaf_hex = compute_sha256(data=chunk)
            leaves.append(bytes.fromhex(leaf_hex))

    if not leaves:
        return 0

    num_leaves = len(leaves)

    # Build the Merkle tree level by level
    tree = []
    current_level = list(leaves)

    while len(current_level) > 1:
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])

        tree.append(list(current_level))

        next_level = []
        for i in range(0, len(current_level), 2):
            combined = current_level[i] + current_level[i + 1]
            parent_hex = compute_sha256(data=combined)
            next_level.append(bytes.fromhex(parent_hex))

        current_level = next_level

    tree.append(current_level)

    merkle_root = tree[-1][0].hex()
    prefix = path.name

    # Export a proof for each chunk
    for chunk_index in range(num_leaves):
        proof = []
        index = chunk_index

        for level in tree[:-1]:
            sibling_index = index ^ 1
            sibling = level[sibling_index]
            is_left = sibling_index < index
            proof.append((sibling.hex(), is_left))
            index //= 2

        output_path = output_dir / f"{prefix}_chunk_{chunk_index}_proof.json"

        # Reuse existing single-proof exporter
        export_merkle_proof(
            proof=proof,
            chunk_index=chunk_index,
            chunk_size=chunk_size,
            merkle_root=merkle_root,
            output_path=output_path
        )

    return num_leaves


def verify_merkle_proof_from_file(
    proof_file_path: Union[str, Path],
    chunk_data: bytes,
    expected_root: str
) -> bool:
    """
    Verify a Merkle proof from a ZKP-compatible JSON file.
    """
    proof_file_path = Path(proof_file_path)

    if not proof_file_path.exists():
        raise FileNotFoundError(f"Proof file not found: {proof_file_path}")

    with proof_file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Malformed proof file: expected JSON object")

    if "public_inputs" not in data or "witness" not in data:
        raise ValueError("Malformed proof file: missing public_inputs or witness")

    public_inputs = data["public_inputs"]
    witness = data["witness"]

    required_public_keys = {"merkle_root", "chunk_index", "chunk_size"}
    if not isinstance(public_inputs, dict) or not required_public_keys.issubset(public_inputs.keys()):
        raise ValueError("Malformed proof file: missing required keys in public_inputs")

    if not isinstance(witness, dict) or "sibling_hashes" not in witness:
        raise ValueError("Malformed proof file: missing sibling_hashes in witness")

    proof_root = public_inputs["merkle_root"]
    if proof_root != expected_root:
        return False

    proof = witness["sibling_hashes"]

    if not isinstance(proof, list):
        raise ValueError("Malformed proof: sibling_hashes must be a list")

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
        raise ValueError(
            "Exactly one of 'data' or 'file_path' must be provided."
        )

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openverifiablellm.utils <input_dump>")
        sys.exit(1)

    logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
    )
    extract_text_from_xml(sys.argv[1])
