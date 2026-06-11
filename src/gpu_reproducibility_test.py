"""Fresh-vs-fresh bitwise reproducibility test on the active device.

Trains the deterministic NanoGPT twice from scratch, with no checkpoint reuse,
and asserts that the two runs produce identical loss curves and bitwise-identical
parameters. On CPU this reproduces the Phase 1 baseline; on a CUDA GPU it is the
Phase 3 claim — that with a pinned cuBLAS workspace and deterministic cuDNN, the
*same* GPU yields the *same bits* run to run.

Run from the ``src`` directory:

    python gpu_reproducibility_test.py

It also appends a short proof block to ``../proofs/device_determinism_log.txt``.
"""

import os
import json
import hashlib
import platform

import torch
import torch.nn.functional as F

from model import TinyGPT
from dataset import TinyDataset
from main import set_seed
from config import TRAIN_CONFIG
from device import get_device, device_name

DEVICE = get_device()


def hash_model(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()


def train_once():
    """One full training run from scratch on DEVICE. Returns (model, losses)."""
    set_seed(TRAIN_CONFIG["seed"])

    dataset = TinyDataset()
    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=TRAIN_CONFIG["embed_dim"],
        num_heads=TRAIN_CONFIG["num_heads"],
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])

    x, y = dataset.get_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    losses = []
    for step in range(TRAIN_CONFIG["total_steps"]):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


def main():
    print(f"\n=== Fresh-vs-fresh determinism on {DEVICE.type.upper()} "
          f"({device_name(DEVICE)}) | torch {torch.__version__} ===")

    model1, losses1 = train_once()
    model2, losses2 = train_once()

    losses_match = losses1 == losses2
    params_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model1.parameters(), model2.parameters())
    )
    hash1, hash2 = hash_model(model1), hash_model(model2)

    print(f"\nFinal loss (run 1): {losses1[-1]:.8f}")
    print(f"Final loss (run 2): {losses2[-1]:.8f}")
    print(f"Loss curves identical: {losses_match}")
    print(f"Bitwise parameter match: {params_match}")
    print(f"Model hash (run 1): {hash1[:16]}...")
    print(f"Model hash (run 2): {hash2[:16]}...")

    ok = losses_match and params_match and (hash1 == hash2)
    if ok:
        print("\n(❁ ´◡`❁) PASSED: same device is bitwise reproducible.")
    else:
        print("\n(╯°□°）╯︵ ┻━┻  FAILED: entropy detected on this device.")

    _write_proof(losses1, losses2, hash1, hash2, ok)
    return ok


def _write_proof(losses1, losses2, hash1, hash2, ok):
    """Append a structured proof block next to the CPU determinism log."""
    proofs_dir = os.path.join(os.path.dirname(__file__), "..", "proofs")
    os.makedirs(proofs_dir, exist_ok=True)
    accelerator_version = None
    if DEVICE.type == "cuda":
        accelerator_version = torch.version.cuda
    elif DEVICE.type == "xpu":
        accelerator_version = getattr(torch.version, "xpu", None)
    record = {
        "device": DEVICE.type,
        "device_name": device_name(DEVICE),
        "torch": torch.__version__,
        "accelerator_version": accelerator_version,
        "os": platform.platform(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "final_loss": losses1[-1],
        "loss_curves_identical": losses1 == losses2,
        "model_hash_run1": hash1,
        "model_hash_run2": hash2,
        "bitwise_reproducible": ok,
    }
    log_path = os.path.join(proofs_dir, "device_determinism_log.txt")
    with open(log_path, "a") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    print(f"\n ~> Proof appended to {os.path.normpath(log_path)}")


if __name__ == "__main__":
    main()
