import os
import sys

import torch
import torch.nn as nn
from utils import hash_file, save_deterministic, set_seed, update_manifest


def main():
    set_seed(99)

    BASE_PATH = "base_checkpoint.pt"
    if not os.path.exists(BASE_PATH):
        print(f"GYAH ERROR: '{BASE_PATH}' not found.")
        print("Please run 'python train_base.py' first to generate the base model.")
        sys.exit(1)

    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    base_ckpt = "base_checkpoint.pt"
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=True))
    base_hash = hash_file(base_ckpt)

    with torch.no_grad():
        for param in model.parameters():
            param.data += 0.001

    ft_ckpt = "finetuned_checkpoint.pt"
    ft_hash = save_deterministic(model.state_dict(), ft_ckpt)

    update_manifest(
        "finetune",
        {
            "base_checkpoint_hash": base_hash,
            "finetune_dataset_hash": "deterministic_shift_0.001",
            "finetune_config": "param_shift",
            "checkpoint_hash": ft_hash,
        },
    )

    print(f"Finetune run hash: {ft_hash}")
    actual_hash = hash_file(ft_ckpt)
    status = "FINE MATCH" if actual_hash == ft_hash else "MISMATCH"
    print(f"Finetune run hash (again): {actual_hash}   {status}")


if __name__ == "__main__":
    main()
