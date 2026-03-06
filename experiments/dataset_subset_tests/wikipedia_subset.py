import random

SEED = 42
random.seed(SEED)

# fake dataset for now
data = [f"Article {i}" for i in range(1000)]

subset = random.sample(data, 20)

print("Wikipedia subset:")

for item in subset:
    print(item)