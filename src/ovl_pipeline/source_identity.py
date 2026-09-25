"""Source identity available before importing the target numerical runtime."""
from pathlib import Path

from .canonical import digest, file_hash


def code_root():
    base = Path(__file__).parent
    return digest([{"path": "ovl_pipeline/" + p.name, "sha256": file_hash(p)}
                   for p in sorted(base.glob("*.py"))]
                  + [{"path": "model.py", "sha256": file_hash(base.parent / "model.py")}])
