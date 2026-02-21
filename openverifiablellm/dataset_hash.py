import hashlib
from pathlib import Path
from typing import Union


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