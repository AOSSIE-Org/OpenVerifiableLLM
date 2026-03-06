import hashlib
import sys
from pathlib import Path


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


checkpoint_path = Path(__file__).parent / "checkpoint.pt"

if not checkpoint_path.exists():
    print("Checkpoint file not found. Run train_tiny_model.py first.")
    sys.exit(1)

hash_value = file_hash(checkpoint_path)

print("Checkpoint SHA256 hash:", hash_value)

second_hash = file_hash(checkpoint_path)

if hash_value != second_hash:
    print("Hash mismatch detected!")
    sys.exit(1)

print("Hash verification successful")