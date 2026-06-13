import json
import hashlib
import torch
import sys
import platform
import os
from pathlib import Path
from dataset import TinyDataset
from config import TRAIN_CONFIG, get_config_hash
from artifacts import (
    CHECKPOINT_MERKLE_PATH,
    CHECKPOINT_STATE_PATH,
    CHECKPOINT_WEIGHTS_PATH,
    build_merkle_manifest,
    compute_sha256,
    hash_json,
)

def hash_dict(d):
    # Sort keys to ensure deterministic JSON stringification
    return hash_json(d)

def generate_global_manifest():
    if not os.path.exists("eval_manifest.json"):
        raise RuntimeError("Missing eval_manifest.json. Please run src/eval.py first to generate the evaluation hashes.")

    print("Generating The Global Verification Manifest...")

    # 1. Environment Fingerprint
    env_fingerprint = {
        "torch": torch.__version__,
        "python": sys.version.split(' ')[0],
        "os": platform.platform()
    }
    env_hash = hash_dict(env_fingerprint)

    # 2. Configuration Hash
    config_hash = get_config_hash()

    # 3. Dataset Hash
    dataset = TinyDataset()
    dataset_hash = hashlib.sha256(dataset.encoded.numpy().tobytes()).hexdigest()

    # 4. Model artifact hash. Prefer safetensors because it is byte-stable;
    # keep the .pt fallback for older runs that only have replay checkpoints.
    model_artifact_path = Path(
        CHECKPOINT_WEIGHTS_PATH
        if os.path.exists(CHECKPOINT_WEIGHTS_PATH)
        else CHECKPOINT_STATE_PATH
    )
    model_artifact = str(model_artifact_path)
    model_hash = compute_sha256(file_path=model_artifact_path)
    if os.path.exists(CHECKPOINT_MERKLE_PATH):
        with open(CHECKPOINT_MERKLE_PATH, "r", encoding="utf-8") as f:
            model_merkle = json.load(f)
    else:
        model_merkle = build_merkle_manifest(model_artifact_path)

    # 5. Eval Manifest Hash (run eval.py before this script)
    with open("eval_manifest.json", "r") as f:
        eval_manifest = json.load(f)
    eval_hash = hash_dict(eval_manifest)

    # 6. Build the Vault
    global_manifest = {
        "1_environment_hash": env_hash,
        "2_training_config_hash": config_hash,
        "3_dataset_hash": dataset_hash,
        "4_model_checkpoint_hash": model_hash,
        "4_model_checkpoint_artifact": model_artifact,
        "4_model_checkpoint_merkle_root": model_merkle["merkle_root"],
        "4_model_checkpoint_chunk_size_bytes": model_merkle["chunk_size_bytes"],
        "4_model_checkpoint_chunk_count": model_merkle["chunk_count"],
        "5_eval_manifest_hash": eval_hash,
    }

    # 7. Seal the Vault
    global_manifest["99_GLOBAL_PIPELINE_HASH"] = hash_dict(global_manifest)

    with open("pipeline_manifest.json", "w") as f:
        json.dump(global_manifest, f, indent=2)

    print("\n Global Manifest Sealed:")
    print(json.dumps(global_manifest, indent=2))

if __name__ == "__main__":
    generate_global_manifest()
