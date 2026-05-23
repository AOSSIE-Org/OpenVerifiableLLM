import json
import os

from utils import get_path, hash_file


def verify():
    if not os.path.exists(get_path("manifest.json")):
        print("Run train_base.py and finetune.py first.")
        return

    with open(get_path("manifest.json"), "r") as f:
        manifest = json.load(f)

    print(" End to End Plipeline Verification \n")

    base_actual = hash_file(get_path("base_checkpoint.pt"))
    base_expected = manifest["base"]["checkpoint_hash"]
    base_match = "BINGO" if base_actual == base_expected else "NUH-UH"
    print(f"Base expected: {base_expected}")
    print(f"Base actual  : {base_actual} {base_match}\n")

    ft_actual = hash_file(get_path("finetuned_checkpoint.pt"))
    ft_expected = manifest["finetune"]["checkpoint_hash"]
    ft_match = "BINGO" if ft_actual == ft_expected else "NUH-UH"
    print(f"Finetune expected: {ft_expected}")
    print(f"Finetune actual  : {ft_actual} {ft_match}")


if __name__ == "__main__":
    verify()
