import time
import os
import tempfile
from openverifiablellm.utils import compute_merkle_root, generate_merkle_proof

def create_dummy_file(filepath, size_mb):
    """Creates a dummy file with random bytes of specified size in MB."""
    with open(filepath, "wb") as f:
        f.write(os.urandom(size_mb * 1024 * 1024))

def run_benchmark():
    print("--- Starting Benchmark ---")
    size_mb = 50
    chunk_size = 1024 * 1024  # 1MB

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        dummy_file_path = temp_file.name

    try:
        print(f"Creating a {size_mb}MB dummy file for benchmarking...")
        create_dummy_file(dummy_file_path, size_mb)

        # Benchmark compute_merkle_root
        start_time = time.perf_counter()
        root_hex = compute_merkle_root(dummy_file_path, chunk_size=chunk_size)
        end_time = time.perf_counter()

        root_time = end_time - start_time
        print(f"compute_merkle_root ({size_mb}MB file): {root_time:.4f} seconds")

        # Benchmark generate_merkle_proof
        start_time = time.perf_counter()
        # Generate proof for middle chunk
        proof = generate_merkle_proof(dummy_file_path, chunk_index=25, chunk_size=chunk_size)
        end_time = time.perf_counter()

        proof_time = end_time - start_time
        print(f"generate_merkle_proof ({size_mb}MB file, chunk 25): {proof_time:.4f} seconds")

        print("--- Benchmark Complete ---")
        return root_time, proof_time

    finally:
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

if __name__ == "__main__":
    run_benchmark()
