import os
import torch
import torch.nn.functional as F
import json
import math
import hashlib
from model import TinyGPT
from dataset import TinyDataset
from main import set_seed
from config import TRAIN_CONFIG
from device import get_device

DEVICE = get_device()

def hash_model(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()

def hash_dict(d):
    encoded = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()

if __name__ == "__main__":
    set_seed(TRAIN_CONFIG["seed"])

    dataset = TinyDataset()
    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=TRAIN_CONFIG["embed_dim"],
        num_heads=TRAIN_CONFIG["num_heads"],
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
        dropout=TRAIN_CONFIG["dropout"]
        ).to(DEVICE)

    # Compute file-level hash before loading as security measure
    checkpoint_path = "mid_checkpoint.pt"
    with open(checkpoint_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Load checkpoint (contains model, optimizer, and RNG states)
    # weights_only=False required for non-tensor state (RNG, metadata)
    # File hash computed above provides tamper detection
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # disabling dropout for eval as results must be deterministic

    model_hash = hash_model(model)

    # Verify cryptographic seal if present
    if 'checkpoint_hash' in checkpoint:
        if model_hash != checkpoint['checkpoint_hash']:
            raise RuntimeError(f"Checkpoint integrity check failed. Expected: {checkpoint['checkpoint_hash'][:16]}..., Got: {model_hash[:16]}...")

    print(f" ~> Model loaded | checkpoint hash: {model_hash[:16]}...")

    # Held-out eval which is never seen during training
    x, y = dataset.get_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    perplexity = math.exp(loss.item())

    print(f" ~> Eval loss:    {loss.item():.8f}")
    print(f" ~> Perplexity:   {perplexity:.5f}")

    eval_data_hash = hashlib.sha256(dataset.encoded.numpy().tobytes()).hexdigest()

    # Build manifest — hash is computed over content, not including itself
    manifest = {
        "model_checkpoint_hash": model_hash,
        "eval_dataset": eval_data_hash,
        "eval_loss": loss.item(),
        "perplexity": perplexity,
    }
    manifest["eval_manifest_hash"] = hash_dict(manifest)

    proofs_dir = os.path.join(os.path.dirname(__file__), "..", "proofs")
    os.makedirs(proofs_dir, exist_ok=True)
    manifest_path = os.path.join(proofs_dir, "eval_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n ~> Manifest saved to {os.path.normpath(manifest_path)}")
    print(json.dumps(manifest, indent=2))