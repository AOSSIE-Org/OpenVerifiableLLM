"""Operator-local publication pause; read-only verification remains available."""
from pathlib import Path
from .canonical import EvidenceError


def require_publication_open():
    marker = Path.home() / ".local/state/openverifiablellm/publication-paused.json"
    if marker.exists() or marker.is_symlink():
        raise EvidenceError("Publication paused for coordinated privacy maintenance")
