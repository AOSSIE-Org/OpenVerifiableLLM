import random

SEED = 42
random.seed(SEED)

data = [f"Article {i}" for i in range(1000)]

# deterministic slice instead of random.sample
subset = data[:20]

print("Wikipedia subset:")

for item in subset:
    print(item)