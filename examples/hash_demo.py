from pathlib import Path
from openverifiablellm.dataset_hash import compute_sha256


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "sample_wiki.txt"

    dataset_hash = compute_sha256(dataset_path)

    print("Dataset Hash:")
    print(dataset_hash)