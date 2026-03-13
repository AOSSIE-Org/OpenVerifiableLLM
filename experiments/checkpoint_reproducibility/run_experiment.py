import hashlib
import os

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

DATA_GEN = torch.Generator().manual_seed(42)
INPUTS = torch.randn(5, 10, generator=DATA_GEN)
TARGETS = torch.randn(5, 2, generator=DATA_GEN)

def train_and_hash(seed, filename):
    torch.manual_seed(seed) 
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    for _ in range(15):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(INPUTS), TARGETS)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), filename)
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

if __name__ == "__main__":
    hash_a1 = train_and_hash(123, "ckpt_a1.pt")
    hash_a2 = train_and_hash(123, "ckpt_a2.pt")
    print(f"Experiment A\nhash1: {hash_a1}...\nhash2: {hash_a2}...")
    print(f"Result: {'deterministic ✔' if hash_a1 == hash_a2 else 'failed ❌'}\n")

    hash_b1 = train_and_hash(123, "ckpt_b1.pt")
    hash_b2 = train_and_hash(999, "ckpt_b2.pt")
    print(f"Experiment B\nhash1: {hash_b1}...\nhash2: {hash_b2}...")
    print(f"Result: {'seed affects training ✔' if hash_b1 != hash_b2 else 'failed ❌'}\n")

    with open("ckpt_a1.pt", "ab") as f:
        f.write(b"tamper_data")
    with open("ckpt_a1.pt", "rb") as f:
        tampered_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    print("Experiment C")
    print("tampered checkpoint detected ✔" if tampered_hash != hash_a1 else "failed ❌")

    for f in ["ckpt_a1.pt", "ckpt_a2.pt", "ckpt_b1.pt", "ckpt_b2.pt"]:
        if os.path.exists(f):
            os.remove(f)