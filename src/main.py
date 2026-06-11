import os
import torch
import numpy as np
import random

# Importing device pins CUBLAS_WORKSPACE_CONFIG before the first CUDA op, which
# deterministic GPU matmuls require. Harmless (no-op) on CPU.
import device  # noqa: F401

def set_seed(seed: int = 99): #never 42
    # Belt-and-suspenders: also set the cuBLAS workspace here in case set_seed is
    # used standalone before `device` is imported elsewhere. Read once at CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed) #this fixes the random module
    np.random.seed(seed) #this fixes the numpy random
    torch.manual_seed(seed) #this fixes the weights (CPU + accelerator host-side seed)

    device.seed_accelerators(seed) #explicitly seed every CUDA/XPU generator (dropout, init)

    # Backend-aware: use_deterministic_algorithms (all backends) + cuDNN flags (CUDA only).
    device.configure_determinism()

    print(f"seed set to {seed}")

if __name__ == "__main__":
    set_seed(99)

    x = torch.randn(3, 3)
    print("Tensor X:") #test randomness
    print(x)

    print("Deterministic enabled:", torch.are_deterministic_algorithms_enabled())

    from dataset import TinyDataset

    dataset = TinyDataset()
    x, y = dataset.get_batch()

    print("Input:", x)
    print("Target:", y)


#for linear Model
#uncomment this block , when running the linear model code blocks
'''
import torch.optim as optim
import torch.nn as nn
# for linear model: from model import TinyModel
from model import TinyGPT
import hashlib

model = TinyGPT(vocab_size=dataset.vocab_size)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

for step in range(5):
    logits = model(x)
    #for linear model: loss = criterion(logits, y[0])
    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss:{loss.item()}")

torch.save(model.state_dict(), "model.pt")

with open("model.pt", "rb") as f:
    model_hash = hashlib.sha256(f.read()).hexdigest()

print(f"FINAL MODEL HASH: {model_hash}")
'''