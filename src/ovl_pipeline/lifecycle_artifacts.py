"""Exact artifact inventories and durable local stage trees; no model imports."""
import os
from pathlib import Path

from .canonical import EvidenceError, confined, inventory, verify_inventory
from .lifecycle import sync_directory


def file_names(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError("artifact root must be a regular directory")
    names = []
    for p in root.rglob("*"):
        if p.is_symlink() or (not p.is_file() and not p.is_dir()):
            raise EvidenceError("unsupported artifact file type")
        if p.is_file():
            names.append(p.relative_to(root).as_posix())
    return sorted(names)


def snapshot(root):
    return {"files": inventory(root, file_names(root))}


def check_snapshot(root, receipt):
    if type(receipt) is not dict or set(receipt) != {"files"}:
        raise EvidenceError("invalid stage receipt")
    verify_inventory(root, receipt["files"])
    if file_names(root) != sorted(e["path"] for e in receipt["files"]):
        raise EvidenceError("stage inventory contains extra or missing artifacts")


def durable_tree(root):
    for name in file_names(root):
        with confined(root, name).open("rb") as f:
            os.fsync(f.fileno())
    # Children first, then parents, before the publication rename/receipt.
    directories = [p for p in Path(root).rglob("*") if p.is_dir()]
    for p in sorted(directories, key=lambda p: len(p.parts), reverse=True) + [Path(root)]:
        sync_directory(p)
