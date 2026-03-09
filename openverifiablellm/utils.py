import bz2
import re
import defusedxml.ElementTree as ET
from pathlib import Path
import sys
import hashlib
import logging
import json
import platform
from typing import Union, Optional, Dict, Any, List, Tuple, Generator, Iterable
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
    output_path: Union[str, Path]
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

def load_merkle_proof(
    proof_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load Merkle proof from a JSON file.
    """
    proof_path = Path(proof_path)

    with proof_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Content before line 270 remains unchanged
# Entire function definition from lines 270-314 should be deleted
def verify_merkle_proof_from_file(
    proof_file_path: Union[str, Path],
    chunk_data: bytes,
    expected_root: str
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

# ---------------------------------------------------------------------------
# Streaming / generator-based variants
# ---------------------------------------------------------------------------

def stream_chunks(
    file_path: Union[str, Path],
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES,
) -> Generator[bytes, None, None]:
    """
    Yield successive raw byte chunks from *file_path* without loading the
    entire file into memory.

    This is the streaming analogue to the chunk-reading loop inside
    ``compute_merkle_root``.  Callers can process each chunk on-the-fly
    (e.g. hash it, write it somewhere) without ever holding more than one
    chunk in RAM at a time.

    Parameters
    ----------
    file_path:
        Path to any binary file (plain or bz2-compressed).
    chunk_size:
        Number of bytes per chunk.  Must be a positive integer.
        Defaults to ``MERKLE_CHUNK_SIZE_BYTES`` (1 MB).

    Yields
    ------
    bytes
        Raw byte chunk of at most *chunk_size* bytes.
        The final chunk may be shorter if the file size is not a multiple
        of *chunk_size*.

    Raises
    ------
    ValueError
        If *chunk_size* is not a positive integer.
    FileNotFoundError
        If *file_path* does not exist.

    Examples
    --------
    >>> for chunk in stream_chunks("data/raw/simplewiki.xml.bz2"):
    ...     process(chunk)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def stream_sha256(
    file_path: Union[str, Path],
    chunk_size: int = 8192,
) -> Generator[Tuple[bytes, str], None, None]:
    """
    Stream SHA-256 hashes of successive chunks from *file_path*.

    Unlike ``compute_sha256``, which returns a **single** hash over the
    entire file, this generator yields ``(chunk_bytes, partial_hex)``
    pairs as each chunk is read.  After the generator is exhausted the
    last ``partial_hex`` value equals the SHA-256 of the whole file.

    Parameters
    ----------
    file_path:
        Path to the file to hash.
    chunk_size:
        Read buffer size in bytes (default 8 192).

    Yields
    ------
    (chunk_bytes, running_hex) : Tuple[bytes, str]
        *chunk_bytes* — the raw bytes just read.
        *running_hex* — SHA-256 hex digest of **all bytes read so far**
        (i.e. the rolling hash after absorbing *chunk_bytes*).

    Examples
    --------
    >>> *_, (_, final_hash) = stream_sha256("data/raw/dump.xml.bz2")
    >>> print(final_hash)   # same value as compute_sha256(file_path=...)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
            yield chunk, sha256.hexdigest()


def compute_merkle_root_streaming(
    chunks: Iterable[bytes],
) -> str:
    """
    Compute a Merkle root over an **arbitrary iterable of byte chunks**
    without requiring a seekable file on disk.

    This is the streaming counterpart to ``compute_merkle_root``.  It
    accepts any iterable — a ``stream_chunks()`` generator, a network
    socket, a list of pre-computed blobs, etc. — so callers are not
    restricted to file-backed data.

    The leaf hashes and tree construction follow the exact same algorithm
    as ``compute_merkle_root`` to ensure identical roots for identical
    content.

    Parameters
    ----------
    chunks:
        Any iterable that yields ``bytes`` objects.  Each object is
        treated as one Merkle leaf.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 Merkle root.
        Returns ``compute_sha256(data=b"")`` for an empty iterable.

    Examples
    --------
    >>> root = compute_merkle_root_streaming(stream_chunks("dump.xml.bz2"))
    >>> assert root == compute_merkle_root("dump.xml.bz2")
    """
    leaves: List[bytes] = []

    for chunk in chunks:
        leaf_hex = compute_sha256(data=chunk)
        leaves.append(bytes.fromhex(leaf_hex))

    if not leaves:
        return compute_sha256(data=b"")

    while len(leaves) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            combined = left + right
            parent_hex = compute_sha256(data=combined)
            next_level.append(bytes.fromhex(parent_hex))
        leaves = next_level

    return leaves[0].hex()


def stream_extract_text_from_xml(
    input_path: Union[str, Path],
) -> Generator[str, None, None]:
    """
    Stream cleaned article texts from a Wikipedia XML dump without writing
    any output file.

    This is the generator-based (streaming) counterpart to
    ``extract_text_from_xml``.  It yields one cleaned plain-text string
    per Wikipedia article, keeping heap usage at O(1) regardless of dump
    size.

    Supports both plain ``.xml`` and bzip2-compressed ``.xml.bz2`` inputs
    by sniffing the first three bytes for the BZh magic header — exactly
    the same auto-detection logic used in ``extract_text_from_xml``.

    Parameters
    ----------
    input_path:
        Path to a Wikipedia XML dump (compressed or uncompressed).

    Yields
    ------
    str
        Cleaned plain-text content of one Wikipedia article.
        Articles that are empty after cleaning are silently skipped.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist on disk.
    ET.ParseError
        If the XML stream is structurally malformed.

    Examples
    --------
    >>> total_chars = sum(len(t) for t in stream_extract_text_from_xml("dump.xml.bz2"))
    >>> print(f"Total characters: {total_chars:,}")
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Dump file not found: {input_path}")

    with open(input_path, "rb") as probe:
        is_bz2 = probe.read(3) == b"BZh"

    open_func = bz2.open if is_bz2 else open

    with open_func(input_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("page"):
                text_elem = elem.find(".//{*}text")
                raw_text: str = ""
                if text_elem is not None and text_elem.text:
                    raw_text = text_elem.text
                elem.clear()

                if not raw_text:
                    continue

                cleaned = clean_wikitext(raw_text)
                if cleaned:
                    yield cleaned

    logger.info("Finished streaming articles from '%s'.", input_path.name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openverifiablellm.utils <input_dump>")
        sys.exit(1)

    logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
    )
    extract_text_from_xml(sys.argv[1])
