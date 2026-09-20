# Candidate checkpoint sizing, CPU only

`measurement.json` records actual local safe-state sizes for the candidate in
`recipe.json`: 23,132,160 distinct trainable parameters, 148,046,771 bytes before
optimizer state exists, and 333,138,942 bytes after one synthetic update. Both
checkpoints were read back through the complete safe-state validator. Timing is
local CPU/storage timing, not GPU throughput or a production cost forecast.

The original command used the repository `.venv`, `PYTHONPATH=src` and
`TOKENIZERS_PARALLELISM=false`. Its numerical operations can be reproduced with
the following recipe (documented after that command, not an additional execution):

```python
from pathlib import Path
import torch
from ovl_pipeline.canonical import read_json
from ovl_pipeline.training import initialize, update
from ovl_pipeline.state import save_state, read_state

recipe = read_json(Path("project/evidence/candidate-state-sizing-v1/recipe.json"))
model, optimizer, control = initialize(recipe)
initial = save_state(Path("fresh-sizing-initial"), model, optimizer, control)
batch = {
    "inputs": torch.arange(512).reshape(1, 512),
    "targets": torch.arange(1, 513).reshape(1, 512),
    "mask": torch.ones((1, 512), dtype=torch.bool),
    "target_ids": torch.arange(512).reshape(1, 512),
}
control = update(model, optimizer, batch, control, 512)
updated = save_state(Path("fresh-sizing-updated"), model, optimizer, control)
read_state(Path("fresh-sizing-initial"), initial)
read_state(Path("fresh-sizing-updated"), updated)
```

Original numerical bytes remain under `.ovllm-cache/candidate-state-sizing-v1/`.
These weights are discarded development material and must never initialize the
production model. Complete GPU recording, replay, checkpoint density, transfer
rates and full corpus counts are still required for the production forecast.
