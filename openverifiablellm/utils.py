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
from typing import Union, Optional

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
def compute_merkle_root(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
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
