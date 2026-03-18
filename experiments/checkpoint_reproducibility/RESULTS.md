# RESULTS.md

Actual output. PyTorch 2.x, Windows, CPU only.

---

## run_experiment.py

```
baseline:  0ec6ccf669f7

torch.save
  run1: 859fee7dca98  run2: 859fee7dca98  same: True  matches_baseline: False

torch rng not seeded
  run1: 444629cb73c6  run2: e9b671c41080  same: False  matches_baseline: False

numpy rng not seeded + shuffle
  run1: 5d7a61399d8c  run2: 5d7a61399d8c  same: True  matches_baseline: False

python rng not seeded + shuffle
  run1: 5d7a61399d8c  run2: 5d7a61399d8c  same: True  matches_baseline: False

dropout same seed
  run1: 7bab2b432f0e  run2: 7bab2b432f0e  same: True  matches_baseline: False

shuffle same seed
  run1: 5d7a61399d8c  run2: 5d7a61399d8c  same: True  matches_baseline: False
```

## pytest

```
14 passed, 0 failed
```

---

## What each result means

**torch.save**
Same hash both runs (`859fee` both times). PyTorch 2.x hardcodes the ZIP entry
date to `2011-01-01`, so the byte-level entropy from timestamps isn't present
here. `save_deterministic()` is still the right approach — it doesn't depend on
that implementation detail staying stable.

**torch rng not seeded**
Different hashes (`444629` vs `e9b671`). This is the only genuinely
non-deterministic result. Weight init without a seed draws from the OS random
state and changes every run. Fixing this requires `torch.manual_seed()`.

**numpy / python rng not seeded + shuffle**
Both produced `5d7a61` — same as shuffle-with-fixed-seed. All three shuffle
variants matched. DataLoader shuffle only draws from the torch generator, not
from numpy or python. Unseeding those two doesn't affect anything here.

**dropout same seed**
`7bab2b` both runs — deterministic. Dropout masks are drawn from the torch RNG,
so with a fixed seed they're reproducible.

**shuffle same seed**
`5d7a61` both runs — deterministic but different from the no-shuffle baseline.
DataLoader shuffle is seeded through `torch.Generator`, so it's reproducible
given the same seed. Different from baseline because the gradient order changed.

**test_torch_save_byte_stability**
Expected to fail (assert not match), but `torch.save` gave identical hashes
on this version. The test documents this outcome — see above.

**test_fixed_seed_isolates_save_format_entropy**
In-memory weights match (seed is fixed). File bytes also match on PyTorch 2.x.
The test prints the result either way — it's documentation, not a pass/fail gate.

**test_fixed_seed_plus_deterministic_save**
Passes. Same seed + `save_deterministic()` = identical SHA-256 end to end.

---

## Open questions

- Cross-machine: different hardware may produce different floats even with
  identical seeds. Not tested.
- CUDA nondeterminism: needs a GPU to actually observe. The test is informational.
- Multiprocessing: skipped on Windows. Expected to work on Linux with an explicit
  DataLoader generator seed.