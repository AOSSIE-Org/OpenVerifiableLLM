import torch
import random
import numpy as np

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# simple tiny model
model = torch.nn.Linear(10, 2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(100):
    x = torch.randn(32, 10)
    y = torch.randn(32, 2)

    loss = ((model(x) - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "checkpoint.pt")

print("Checkpoint saved as checkpoint.pt")