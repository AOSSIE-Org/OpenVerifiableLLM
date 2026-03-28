import hashlib
import json
import os

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    """Returns the absolute path for a file in this experiment directory."""
    return os.path.join(SCRIPT_DIR, filename)


def set_seed(seed=42):
    """Locks in deterministic behavior."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def hash_file(filepath):
    """Generates a SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def save_deterministic(state_dict, path):
    """Saves state dict without zip metadata to ensure identical hashes."""
    torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    return hash_file(path)


def update_manifest(stage, data, manifest_name="manifest.json"):
    """Appends stage data to the verification manifest."""
    manifest_path = get_path(manifest_name)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    manifest[stage] = data

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
