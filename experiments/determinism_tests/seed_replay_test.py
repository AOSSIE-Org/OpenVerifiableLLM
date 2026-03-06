import torch
from experiments.utils import set_seed

set_seed(42)
x = torch.randn(5)

set_seed(42)
y = torch.randn(5)

if not torch.equal(x, y):
    raise SystemExit("Seed replay mismatch")

print("Generated tensor:", x)