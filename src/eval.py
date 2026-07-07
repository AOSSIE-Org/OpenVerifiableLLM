import hashlib
import json
import math

import torch
import torch.nn.functional as F

from config import model_config
from dataset import get_dataset
from device import get_device
from main import set_seed
from model import build_model
from signing import verified_torch_load
from artifacts import (
    CHECKPOINT_STATE_PATH,
    CHECKPOINT_WEIGHTS_PATH,
    RUNS_DIR,
    hash_json,
    load_model_safetensors,
    model_parameters_sha256,
)

DEVICE = get_device()
MODEL_NAME = "gpt10m"
DATASET_NAME = "shakespeare"
CFG = model_config(MODEL_NAME)


def hash_model(model):
    return model_parameters_sha256(model)


def hash_dict(d):
    return hash_json(d)


if __name__ == "__main__":
    set_seed(CFG["seed"])

    dataset = get_dataset(DATASET_NAME, block_size=CFG["block_size"])
    model = build_model(MODEL_NAME, dataset.vocab_size, CFG).to(DEVICE)

    try:
        # Byte-stable safetensors is the preferred artifact (no pickle, no code-exec).
        load_model_safetensors(model, CHECKPOINT_WEIGHTS_PATH, device=DEVICE)
        checkpoint_source = CHECKPOINT_WEIGHTS_PATH.name  # name only: keep manifests machine-portable
    except FileNotFoundError:
        # Fallback to the replay checkpoint -- but verify the signature BEFORE
        # deserializing (never torch.load(weights_only=False) on unverified bytes).
        checkpoint = verified_torch_load(CHECKPOINT_STATE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        checkpoint_source = CHECKPOINT_STATE_PATH.name
    model.eval()  # disable dropout for eval; results must be deterministic

    model_hash = hash_model(model)
    print(f" ~> Model loaded from {checkpoint_source} | checkpoint hash: {model_hash[:16]}...")

    # Held-out eval batch
    x, y = dataset.get_batch(CFG["batch_size"], CFG["block_size"], device=DEVICE)

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    perplexity = math.exp(loss.item())

    print(f" ~> Eval loss:    {loss.item():.8f}")
    print(f" ~> Perplexity:   {perplexity:.5f}")

    eval_data_hash = hashlib.sha256(dataset.encoded.numpy().tobytes()).hexdigest()

    manifest = {
        "model_checkpoint_hash": model_hash,
        "model_checkpoint_source": checkpoint_source,
        "eval_dataset": eval_data_hash,
        "eval_loss": loss.item(),
        "perplexity": perplexity,
    }
    manifest["eval_manifest_hash"] = hash_dict(manifest)

    eval_manifest_path = RUNS_DIR / "eval_manifest.json"
    eval_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n ~> Manifest saved to {eval_manifest_path}")
    print(json.dumps(manifest, indent=2))
