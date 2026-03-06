import torch
import random
import numpy as np
from pathlib import Path
from experiments.utils import set_seed

SEED = 42


set_seed(SEED)

# simple tiny model
model = torch.nn.Linear(10, 2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for _ in range(100):
    x = torch.randn(32, 10)
    y = torch.randn(32, 2)

    loss = ((model(x) - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

checkpoint_path = Path(__file__).parent / "checkpoint.pt"

torch.save(model.state_dict(), checkpoint_path)

print(f"Checkpoint saved as {checkpoint_path}")