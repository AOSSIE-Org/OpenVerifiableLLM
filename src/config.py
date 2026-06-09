import json
import hashlib

# The Immutable Training Configuration
TRAIN_CONFIG = {
    "embed_dim": 16,
    "num_heads": 2,
    "max_seq_len": 32,
    "dropout": 0.1,
    "lr": 0.01,
    "optimizer": "Adam",
    "seed": 99,
    "total_steps": 10,
    "checkpoint_step": 5
}

def get_config_hash():
    """Returns a deterministic SHA-256 hash of the configuration dict."""
    encoded = json.dumps(TRAIN_CONFIG, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()