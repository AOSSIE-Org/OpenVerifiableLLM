import time
import os
import sys
from openverifiablellm.utils import compute_merkle_root, generate_merkle_proof

def run_benchmark(file_path):
    print("--- Starting Benchmark ---")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    chunk_size = 1024 * 1024  # 1MB

    print(f"Benchmarking file: {file_path}")
    print(f"File size: {size_mb:.2f} MB")

    try:
        # Benchmark compute_merkle_root
        start_time = time.perf_counter()
        root_hex = compute_merkle_root(file_path, chunk_size=chunk_size)
        end_time = time.perf_counter()

        root_time = end_time - start_time
        print(f"compute_merkle_root ({size_mb:.2f} MB file): {root_time:.4f} seconds")
        print(f"Merkle Root: {root_hex}")

        # Benchmark generate_merkle_proof
        start_time = time.perf_counter()
        # Generate proof for a chunk (e.g. chunk 10 if there are enough, otherwise chunk 0)
        chunk_index = 10 if size_mb > 10 else 0
        proof = generate_merkle_proof(file_path, chunk_index=chunk_index, chunk_size=chunk_size)
        end_time = time.perf_counter()

        proof_time = end_time - start_time
        print(f"generate_merkle_proof ({size_mb:.2f} MB file, chunk {chunk_index}): {proof_time:.4f} seconds")

        print("--- Benchmark Complete ---")
        return root_time, proof_time

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_custom.py <path_to_file>")
        sys.exit(1)
    run_benchmark(sys.argv[1])
