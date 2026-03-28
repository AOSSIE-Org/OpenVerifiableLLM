import torch
import torch.nn as nn
from utils import hash_file, save_deterministic, set_seed, update_manifest


def main():
    set_seed(99)

    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    X = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

    ckpt_path = "base_checkpoint.pt"
    ckpt_hash = save_deterministic(model.state_dict(), ckpt_path)

    update_manifest(
        "base", {"seed": 99, "dataset_hash": "synthetic_16x10_seed99", "checkpoint_hash": ckpt_hash}
    )

    print(f"Base run hash: {ckpt_hash}")
    print(f"Base run hash (again): {hash_file(ckpt_path)}   CORRECT MATCH")


if __name__ == "__main__":
    main()
