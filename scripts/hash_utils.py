import hashlib
import sys

def sha256_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

if __name__ == "__main__":
    path = sys.argv[1]
    print(sha256_file(path))