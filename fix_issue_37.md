### PR Description

This pull request introduces an `experiments/` directory into the repository to facilitate testing and validation of determinism and verification assumptions. The new directory includes subdirectories and scripts aligning with the proposed structure in the GitHub issue. Each script is designed to execute small, controlled experiments, enabling verification of assumptions such as checkpoint reproducibility, consistent dataset sampling, and deterministic behavior across environments.

The following structure is implemented in the `experiments/` directory:

```
experiments/
    |-------> checkpoint_reproducibility/
    |               |----> train_tiny_model.py
    |               |----> verify_hash_consistency.py
    |
    |-------> dataset_subset_validation/
    |               |-----> wikipedia_subset.py
    |
    |-------> determinism_tests/
                    |------> seed_replay_test.py
```

### 1. Checkpoint Reproducibility

#### 1.1 `train_tiny_model.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train():
    torch.manual_seed(0)
    model = nn.Linear(10, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Generating some dummy data
    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 1)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    for epoch in range(1):  # Single epoch
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

    checkpoint_path = 'checkpoint.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Model checkpoint saved to {checkpoint_path}')

if __name__ == '__main__':
    train()
```

#### 1.2 `verify_hash_consistency.py`

```python
import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

checkpoint_path = 'checkpoint.pth'
print(f'Checkpoint MD5 hash: {get_file_hash(checkpoint_path)}')
```

### 2. Dataset Subset Validation

#### 2.1 `wikipedia_subset.py`

```python
import random

def sample_wikipedia_subset(seed=0):
    random.seed(seed)
    # Dummy Wikipedia excerpts for testing
    wiki_articles = [
        "Wikipedia is a free online encyclopedia.",
        "It was launched on January 15, 2001.",
        "The name Wikipedia is a portmanteau of the words wiki and encyclopedia.",
        "It is hosted by the Wikimedia Foundation."
    ]
    subset_size = 2
    return random.sample(wiki_articles, subset_size)

if __name__ == '__main__':
    subset = sample_wikipedia_subset()
    print(f'Sampled Wikipedia subset: {subset}')
```

### 3. Determinism Tests

#### 3.1 `seed_replay_test.py`

```python
import random

def run_deterministic_process(seed=0):
    random.seed(seed)
    return [random.random() for _ in range(5)]

if __name__ == '__main__':
    first_run = run_deterministic_process()
    second_run = run_deterministic_process()
    assert first_run == second_run, "Non-deterministic output encountered!"
    print('Deterministic test passed!')
```

### Testing

To test these scripts:

- **Checkpoint reproducibility**: Run `train_tiny_model.py` to create a model checkpoint and then use `verify_hash_consistency.py` to print and compare checkpoint hashes.
- **Dataset subset validation**: Execute `wikipedia_subset.py` with fixed seeds to ensure reproducible subsets.
- **Determinism tests**: Run `seed_replay_test.py` to confirm deterministic outputs across repeated executions.

These experiments will guide future architectural decisions and help ensure end-to-end verifiability in the pipeline.