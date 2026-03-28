import hashlib
import json
import os
import random

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def set_seed(seed=99):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_default_device("cpu")


def hash_file(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def save_deterministic(state_dict, path):
    torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    return hash_file(path)


def update_manifest(stage, data, manifest_path="manifest.json"):
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    manifest[stage] = data

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
