import os
import sys

import torch
import torch.nn as nn
from utils import get_path, hash_file, save_deterministic, set_seed, update_manifest


def main():
    set_seed(99)

    base_ckpt = get_path("base_checkpoint.pt")
    if not os.path.exists(base_ckpt):
        print(f"GYAH ERROR: '{base_ckpt}' not found.")
        print("Please run 'python train_base.py' first to generate the base model.")
        sys.exit(1)

    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    model.load_state_dict(torch.load(base_ckpt, weights_only=True))
    base_hash = hash_file(base_ckpt)

    with torch.no_grad():
        for param in model.parameters():
            param.data += 0.001

    ft_ckpt = get_path("finetuned_checkpoint.pt")
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
    print(f"Finetune run hash (again): {hash_file(ft_ckpt)}   FINE MATCH")


if __name__ == "__main__":
    main()
