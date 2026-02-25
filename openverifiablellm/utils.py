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

def compute_directory_hash(directory_path: Union[str, Path]) -> str:
    """
    Compute a deterministic SHA256 hash of a directory.

    The hash incorporates:
    - Relative file paths (normalized to POSIX style)
    - File contents (via compute_sha256)

    Hidden files and files inside hidden directories (starting with '.')
    are ignored.

    Parameters
    ----------
    directory_path : Union[str, Path]
        Path to directory to hash.

    Returns
    -------
    str
        SHA256 hex digest representing directory contents.
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(directory)

    if not directory.is_dir():
        raise NotADirectoryError(directory)

    sha256 = hashlib.sha256()

    entries = []

    for f in directory.rglob("*"):
        if not f.is_file():
            continue

        relative = f.relative_to(directory)

        if any(part.startswith(".") for part in relative.parts):
            continue

        rel_posix = relative.as_posix()
        entries.append((rel_posix, f))

    entries_sorted = sorted(entries, key=lambda e: e[0])

    for relative_path, file in entries_sorted:

        # Include relative path in hash
        sha256.update(relative_path.encode("utf-8"))
        sha256.update(b"\0")  # delimiter

        # Include file content hash
        file_hash = compute_sha256(file)
        sha256.update(file_hash.encode("utf-8"))
        sha256.update(b"\0")  # delimiter

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
