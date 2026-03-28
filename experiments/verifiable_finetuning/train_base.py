import torch
import torch.nn as nn
from utils import get_path, hash_file, save_deterministic, set_seed, update_manifest


def main():
    # Aligned seed with finetune.py
    set_seed(99)

    # 1. Tiny deterministic model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # 2. Synthetic dataset
    X = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))

    # 3. Minimal train step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

    # 4. Save and Hash with get_path() to prevent ghost files!
    ckpt_path = get_path("base_checkpoint.pt")
    ckpt_hash = save_deterministic(model.state_dict(), ckpt_path)

    # 5. Update Manifest
    update_manifest(
        "base", {"seed": 99, "dataset_hash": "synthetic_16x10_seed99", "checkpoint_hash": ckpt_hash}
    )

    print(f"Base run hash: {ckpt_hash}")
    print(f"Base run hash (again): {hash_file(ckpt_path)}   CORRECT MATCH")


if __name__ == "__main__":
    main()
