#!/usr/bin/env python3
import sys
import json
import hashlib
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openverifiablellm import compute_sha256


def generate_manifest(directory_path: Path) -> dict:
    files = [f for f in directory_path.glob("**/*") if f.is_file()]
    files.sort(key=lambda x: str(x.relative_to(directory_path)).replace("\\", "/"))
    
    manifest_entries = []
    for file in files:
        rel_path = str(file.relative_to(directory_path)).replace("\\", "/")
        file_hash = compute_sha256(file)
        manifest_entries.append({
            "path": rel_path,
            "sha256": file_hash,
            "size": file.stat().st_size
        })
    return {"files": manifest_entries}


def get_manifest_hash(manifest: dict) -> str:
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def create_test_dataset(directory: Path, seed: int = 42) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dataset_files = {
        "file1.txt": f"Test file 1 generated with seed {seed}",
        "file2.txt": f"Test file 2 generated with seed {seed}",
        "metadata.json": json.dumps({"seed": seed, "version": "1.0.0"}, indent=2),
        "subdir/file3.txt": f"Nested file generated with seed {seed}",
    }
    for rel_path, content in dataset_files.items():
        file_path = directory / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def test_reproducibility_multiple_runs(runs: int = 3, seed: int = 42) -> Tuple[bool, List[str], str]:
    hashes = []
    for i in range(runs):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / f"dataset_{i}"
            create_test_dataset(dataset_dir, seed)
            manifest = generate_manifest(dataset_dir)
            root_hash = get_manifest_hash(manifest)
            hashes.append(root_hash)
            print(f"Run {i+1}/{runs}: Hash = {root_hash}")
    all_match = all(h == hashes[0] for h in hashes)
    return all_match, hashes, hashes[0] if hashes else ""


def run_all_tests() -> Dict[str, bool]:
    print("=" * 60)
    print("Running Reproducibility Tests")
    print("=" * 60)
    results = {}
    
    print("\n1. Testing reproducibility across multiple runs...")
    success, hashes, final_hash = test_reproducibility_multiple_runs()
    results["reproducibility"] = success
    print(f"   {'PASSED' if success else 'FAILED'}: All runs produced {'same' if success else 'different'} hash")
    
    print("\n" + "=" * 60)
    if all(results.values()):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run reproducibility tests")
    parser.add_argument("--test", choices=["all", "reproducibility"], default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-hash", action="store_true")
    args = parser.parse_args()
    
    if args.test == "all":
        results = run_all_tests()
        if args.output_hash:
            success, hashes, final_hash = test_reproducibility_multiple_runs(args.runs, args.seed)
            if success:
                print(f"REPRODUCIBLE_HASH={final_hash}")
                with open("reproducible_hash.txt", "w") as f:
                    f.write(final_hash)
        sys.exit(0 if all(results.values()) else 1)
    
    elif args.test == "reproducibility":
        success, hashes, final_hash = test_reproducibility_multiple_runs(args.runs, args.seed)
        if args.output_hash and success:
            print(f"REPRODUCIBLE_HASH={final_hash}")
            with open("reproducible_hash.txt", "w") as f:
                f.write(final_hash)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
