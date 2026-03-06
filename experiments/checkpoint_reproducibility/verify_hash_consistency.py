import sys

hash_value = file_hash("checkpoint.pt")

print("Checkpoint SHA256 hash:", hash_value)

# verify hash consistency
second_hash = file_hash("checkpoint.pt")

if hash_value != second_hash:
    print("Hash mismatch detected!")
    sys.exit(1)

print("Hash verification successful")