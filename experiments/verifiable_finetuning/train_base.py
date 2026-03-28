import torch
import torch.nn as nn
from utils import get_path, hash_file, save_deterministic, set_seed, update_manifest


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

    ckpt_path = get_path("base_checkpoint.pt")
    ckpt_hash = save_deterministic(model.state_dict(), ckpt_path)

    dataset_bytes = X.detach().cpu().numpy().tobytes() + y.detach().cpu().numpy().tobytes()
    dataset_hash = __import__("hashlib").sha256(dataset_bytes).hexdigest()
    update_manifest(
        "base", {"seed": 99, "dataset_hash": dataset_hash, "checkpoint_hash": ckpt_hash}
    )

    print(f"Base run hash: {ckpt_hash}")
    actual_hash = hash_file(ckpt_path)
    status = "CORRECT MATCH" if actual_hash == ckpt_hash else "MISMATCH"
    print(f"Base run hash (again): {actual_hash}   {status}")


if __name__ == "__main__":
    main()
