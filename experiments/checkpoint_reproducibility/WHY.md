# WHY.md

## The original question

In a Discord discussion someone asked whether you could verify a training run just
by hashing the checkpoint. The idea being: re-run training on the same data and
if the hashes match, nothing was hidden.

That only works if training is actually deterministic. I wasn't sure it was, so I
ran some experiments to find out.

---

## What I checked and why

**Does the same seed give the same weights?**
First thing to confirm. If weight init isn't reproducible nothing else will be.
It is — that's `test_same_seed_same_weights`, and it's the floor everything else
builds on.

**Does "same weights" mean "same file"?**
Not automatically. `torch.save()` uses Python's zipfile internally, and ZIP
entries can embed timestamps. If the timestamp changes between saves, the SHA-256
hash changes even though the model is identical. The fix is to write raw tensor
bytes directly with no container — which is what `save_deterministic()` does, and
what safetensors was designed for.

What I didn't expect: on this machine `torch.save()` produced the same hash both
times. Looking into it, PyTorch 2.x hardcodes the ZIP entry date to `2011-01-01`.
So the timestamp problem is already fixed in recent versions — but it's an
implementation detail, not a guarantee, so `save_deterministic()` is still the
right approach.

**Which RNG families actually matter?**
I assumed I needed to seed everything — torch, numpy, python. Turned out numpy
and python are inert for a pure PyTorch training loop. The broad survey confirmed
it: unseeding numpy or python (even with shuffle active) produced the same hash as
the seeded version. The torch seed is the only one that's load-bearing right now.

The numpy and python calls in `set_seed()` are still worth keeping — if the
pipeline ever adds numpy-based augmentation they'll matter — but right now they're
not doing anything.

**Are independent sources actually independent?**
`test_fixed_seed_isolates_save_format_entropy` checks this. Fix the seed so
weights are deterministic, then see if `torch.save()` still produces different
bytes. If they diverge, the two sources are independent. On PyTorch 2.x both
match, which confirms independence from the other direction.

---

## Things that went wrong

The original test for non-contiguous tensors asserted that calling `.numpy()` on
`weight.T` would raise a `RuntimeError`. That used to be true in older NumPy but
isn't anymore — it handles strided arrays silently. Caught on first run, rewrote
it to check byte layout consistency instead.

The first version of `run_experiment.py` created the dataset at module level with
no `if __name__ == "__main__"` guard. On Windows, DataLoader with `num_workers > 0`
spawns workers that re-import the script, which deadlocked immediately. Fixed by
moving dataset creation into `make_dataset()`.

---

## What this doesn't cover

- **Cross-machine**: different CPU architectures can produce slightly different
  floating-point results even with the same seed. Not tested here.
- **CUDA**: the nondeterministic mode test is informational only — the test
  machine is CPU-only. On CUDA some operations are nondeterministic by default.
- **Distributed training**: gradient averaging across workers introduces ordering
  issues that are hard to control. Out of scope.