import hashlib
import sys

def sha256_file(filepath):
    sha256 = hashlib.sha256()
    chunk_size = 1024 * 1024  # 1 MiB for better throughput
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hash_utils.py <filepath>")
        sys.exit(1)
    path = sys.argv[1]
    print(sha256_file(path))