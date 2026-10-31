"""Export a released model to safetensors, without changing any weight.

    python simple/export_safetensors.py --run RUN --out conversation.safetensors
    python simple/export_safetensors.py --run RUN --model base --out base.safetensors

safetensors stores raw tensors without pickle, so loading a file runs no code. The tied embedding and
output weights are stored as two copies, so the file holds exactly the state dict that training wrote,
and `ovl.py verify --weights FILE.safetensors` matches the same chain entry as the .pt file. The
metadata key "ovl" holds a JSON object with the .pt file's name and SHA-256 and the model digest. It is
one key, so the file is byte-identical every time it is exported. Needs `pip install safetensors`.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ovl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="conversation", choices=["base", "conversation"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source = Path(a.run) / f"{a.model}-model.pt"
    if source.resolve() == Path(a.out).resolve():
        p.error("--out must not overwrite the source checkpoint")
    sd = torch.load(source, map_location="cpu", weights_only=True)
    digest = ovl.model_digest(sd)
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()}, a.out,
              metadata={"ovl": json.dumps({"source": source.name, "model_digest": digest,
                                           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                                          sort_keys=True)})
    if ovl.model_digest(load_file(a.out)) != digest:
        sys.exit(f"{a.out}: the weights differ from {source}")
    print(f"{a.out}: model digest {digest}")


if __name__ == "__main__":
    main()
