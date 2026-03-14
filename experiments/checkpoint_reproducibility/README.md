# Checkpoint Reproducibility Experiment

Checks whether identical PyTorch training runs produce identical checkpoints.
This came up in Discord — the question was whether you could verify a training
run by hashing the checkpoint. Before that's useful you need to know if training
is actually deterministic.

Short answer: yes, but only if you seed the RNG and use a deterministic save
format. These experiments figure out which things break it and which don't.

---

## Files

```
run_experiment.py       broad pass - toggles one thing at a time, prints hashes
test_entropy_sources.py one variable per test, run with pytest
utils.py                set_seed() and hash_file()
WHY.md                  what questions I was trying to answer
RESULTS.md              what the results mean
SOURCES.md         sources
```

---

## Run it

```bash
cd experiments/checkpoint_reproducibility

python run_experiment.py

pytest test_entropy_sources.py -v
```

Expected: `run_experiment.py` shows one genuinely non-deterministic result
(unseeded torch RNG). Pytest is 14 passed, 0 failed.

---

## Windows note

`num_workers > 0` is skipped automatically on Windows. PyTorch uses `spawn` for
DataLoader workers which re-imports the script in each process. Making it work
needs more setup than the experiment is worth. See RESULTS.md for expected Linux
behaviour.