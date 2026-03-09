"""
manifest_chain.py
=================
Cryptographically linked manifest chains for tamper detection.

This module provides utilities to create and verify a chain of dataset manifests,
where each manifest includes the SHA256 hash of its predecessor. This forms a
tamper-evident chain: if any manifest in the sequence is modified, the hash
stored in the next manifest no longer matches, and the tampering is immediately
visible (analogous to a wax seal chain).

Usage
-----
# Generate parent hash before writing a new manifest
    parent_hash = get_parent_manifest_hash(manifest_path)
    manifest = { ... , "parent_manifest_hash": parent_hash }

# Verify the chain between two consecutive manifests
    is_valid = verify_manifest_chain_link(previous_manifest_path, current_manifest)

# Verify entire chain from root
    report = verify_manifest_chain(current_manifest_path)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def _canonical_json(obj: Any) -> str:
    """
    Serialize object into canonical JSON format.
    Ensures stable hashing across runs regardless of key order.

    Parameters
    ----------
    obj : Any
        JSON-serializable object

    Returns
    -------
    str
        Canonical JSON string with sorted keys
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_manifest_hash(manifest: Union[str, Path, Dict[str, Any]]) -> str:
    """
    Compute SHA-256 hash of a manifest.

    Can accept:
    - Dict: manifest data object (will be canonical-JSON serialized)
    - str/Path: path to manifest JSON file (will be read and parsed)

    Parameters
    ----------
    manifest : Union[str, Path, Dict[str, Any]]
        Either a manifest dict or a path to a manifest JSON file

    Returns
    -------
    str
        SHA-256 hash in hexadecimal format

    Raises
    ------
    FileNotFoundError
        If manifest is a path and the file does not exist
    ValueError
        If manifest JSON is malformed
    """
    if isinstance(manifest, dict):
        manifest_data = manifest
    else:
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            try:
                manifest_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed manifest JSON: {e}")

    # Hash the entire manifest so descendants authenticate both the
    # manifest contents and its declared predecessor.
    hashable = manifest_data.copy()

    canonical = _canonical_json(hashable)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_parent_manifest_hash(
    manifest_path: Union[str, Path],
) -> str:
    """
    Read the SHA256 hash of the manifest at manifest_path, to be stored
    as parent_manifest_hash in the next (replacement) manifest.

    If the file does not exist yet (first run), returns an empty string.

    Parameters
    ----------
    manifest_path : Union[str, Path]
        Path to the manifest file

    Returns
    -------
    str
        SHA-256 hash of the existing manifest, or "" if it doesn't exist
    """
    path = Path(manifest_path)

    if not path.exists():
        logger.info("No previous manifest found at %s — parent_manifest_hash will be empty", path)
        return ""

    try:
        parent_hash = compute_manifest_hash(path)
        logger.info("Parent manifest hash computed: %s", parent_hash)
        return parent_hash
    except Exception as e:
        logger.exception("Could not compute parent manifest hash: %s", e)
        raise


def verify_manifest_chain_link(
    previous_manifest: Union[str, Path, Dict[str, Any]],
    current_manifest: Union[str, Path, Dict[str, Any]],
) -> bool:
    """
    Verify that current_manifest correctly references previous_manifest
    via its parent_manifest_hash field.

    This checks a single link in the chain. If the link is broken, it indicates
    that either:
    - The previous manifest was tampered with/regenerated
    - The current manifest's parent_manifest_hash was modified

    Parameters
    ----------
    previous_manifest : Union[str, Path, Dict[str, Any]]
        The previous manifest (dict or path to file)
    current_manifest : Union[str, Path, Dict[str, Any]]
        The current manifest (dict or path to file)

    Returns
    -------
    bool
        True if parent_manifest_hash in current matches the hash of previous

    Raises
    ------
    FileNotFoundError
        If any required file does not exist
    ValueError
        If manifest JSON is malformed
    """
    # Load current manifest if needed
    if isinstance(current_manifest, dict):
        current_data = current_manifest
    else:
        current_path = Path(current_manifest)
        if not current_path.exists():
            raise FileNotFoundError(f"Current manifest not found: {current_path}")
        with current_path.open("r", encoding="utf-8") as f:
            try:
                current_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed current manifest JSON: {e}")

    # Get stored parent hash
    stored_parent_hash = current_data.get("parent_manifest_hash", "")

    # Compute hash of previous manifest
    expected_parent_hash = compute_manifest_hash(previous_manifest)

    # Compare
    match = stored_parent_hash == expected_parent_hash

    if match:
        logger.info("Manifest chain link verified ✓")
    else:
        logger.error(
            "Manifest chain link broken! ✗\n"
            "  stored (in current)  : %s\n"
            "  computed (from prev) : %s",
            stored_parent_hash,
            expected_parent_hash,
        )

    return match


def verify_manifest_chain(
    manifest_path: Union[str, Path],
    previous_manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Verify the manifest chain up to the given manifest.

    If previous_manifest_path is provided, checks the link between previous
    and current. If not provided, checks that:
    - parent_manifest_hash exists (for current run > 1)
    - parent_manifest_hash is non-empty (indicates there was a previous run)

    Parameters
    ----------
    manifest_path : Union[str, Path]
        Path to the manifest to verify
    previous_manifest_path : Optional[Union[str, Path]]
        Path to the previous manifest (optional, for explicit link verification)

    Returns
    -------
    Dict[str, Any]
        Report with keys:
        - "chain_valid": bool - whether the chain is intact
        - "chain_message": str - human-readable message
        - "has_parent_hash_field": bool - whether parent_manifest_hash field exists
        - "parent_hash_value": str - value of parent_manifest_hash (or "")
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return {
            "chain_valid": False,
            "chain_message": f"Manifest not found: {manifest_path}",
            "has_parent_hash_field": False,
            "parent_hash_value": "",
        }

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as e:
        return {
            "chain_valid": False,
            "chain_message": f"Failed to read manifest: {e}",
            "has_parent_hash_field": False,
            "parent_hash_value": "",
        }

    has_field = "parent_manifest_hash" in manifest_data
    parent_hash_value = manifest_data.get("parent_manifest_hash", "")

    # If explicit previous manifest is provided, verify the link
    if previous_manifest_path is not None:
        try:
            link_valid = verify_manifest_chain_link(previous_manifest_path, manifest_data)
            message = (
                "Chain link verified against previous manifest ✓"
                if link_valid
                else "Chain link broken — previous manifest does not match stored hash ✗"
            )
        except (OSError, ValueError) as exc:
            link_valid = False
            message = f"Failed to verify previous manifest: {exc}"
        return {
            "chain_valid": link_valid,
            "chain_message": message,
            "has_parent_hash_field": has_field,
            "parent_hash_value": parent_hash_value,
        }

    # Otherwise, just check that the field exists (indicating awareness of chain concept)
    message = ""
    if not has_field:
        message = "parent_manifest_hash field missing (may be old manifest)"
        chain_valid = False
    elif parent_hash_value == "":
        message = "parent_manifest_hash is empty (first run in chain)"
        chain_valid = True
    else:
        message = (
            "Cannot verify non-root manifest without previous_manifest_path"
        )
        chain_valid = False

    return {
        "chain_valid": chain_valid,
        "chain_message": message,
        "has_parent_hash_field": has_field,
        "parent_hash_value": parent_hash_value,
    }
