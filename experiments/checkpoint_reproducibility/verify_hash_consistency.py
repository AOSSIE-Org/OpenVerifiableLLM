import hashlib

def file_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)

    return sha.hexdigest()

hash_value = file_hash("checkpoint.pt")

print("Checkpoint SHA256 hash:", hash_value)