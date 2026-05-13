"""
_hashing.py
===========
Shared low-level hashing utilities used across the openverifiablellm package.

These are intentionally kept small and dependency-free so they can be safely
imported from any module without risk of circular imports.
"""

import json
from typing import Any


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
