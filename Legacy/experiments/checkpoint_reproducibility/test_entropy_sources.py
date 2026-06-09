# test_entropy_sources.py
#
# Structured as a series of questions. Each question tries the obvious approach
# first, watches it fail, then works toward a fix. xfail marks a test that is
# *expected* to fail - it shows the broken approach and why, before the working
# version follows.
#
# pytest test_entropy_sources.py -v

import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import hash_file  # noqa: E402

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


g = torch.Generator().manual_seed(99)
INPUTS = torch.randn(5, 10, generator=g)
TARGETS = torch.randn(5, 2, generator=g)


def save_deterministic(state_dict, path):
    with open(path, "wb") as f:
        for k, v in state_dict.items():
            f.write(k.encode())
            f.write(v.detach().cpu().contiguous().numpy().tobytes())


def train(
    *,
    torch_seed=None,
    numpy_seed=None,
    python_seed=None,
    deterministic=True,
    inputs=None,
    targets=None,
    steps=15,
):
    x = INPUTS if inputs is None else inputs
    y = TARGETS if targets is None else targets

    if python_seed is not None:
        random.seed(python_seed)
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(steps):
        opt.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()

    return model.state_dict()


# ===========================================================================
# Question 1: Is PyTorch training deterministic at all?
#
# Naive approach: just run it twice and compare.
# Expected: fails. Weight init draws from the OS random state without a seed.
# Fix: torch.manual_seed()
# ===========================================================================


@pytest.mark.xfail(reason="no seed - weight init draws from OS random state each run")
def test_q1_naive_no_seed_weights_match():
    # First attempt: just run twice, hope for the best
    sd1 = train(torch_seed=None)
    sd2 = train(torch_seed=None)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_q1_seed_fixes_weight_determinism():
    # Add torch.manual_seed() - now weight init is reproducible
    sd1 = train(torch_seed=42)
    sd2 = train(torch_seed=42)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_q1_different_seeds_give_different_weights():
    # Confirm the seed is actually controlling something, not being ignored
    sd1 = train(torch_seed=42)
    sd2 = train(torch_seed=99)
    assert not all(torch.equal(sd1[k], sd2[k]) for k in sd1)


# ===========================================================================
# Question 2: Does "same weights" mean "same file hash"?
#
# You'd assume yes. Turns out torch.save() wraps output in a ZIP archive
# which can embed timestamps. If the timestamp changes, the hash changes
# even though the model is identical.
#
# Naive approach: torch.save() with seed.
# On older PyTorch: fails - ZIP timestamp differs between saves.
# On PyTorch 2.x: unexpectedly passes - ZIP date is hardcoded to 2011-01-01.
# Fix regardless: write raw bytes directly, no container.
# ===========================================================================


@pytest.mark.xfail(
    reason="torch.save embeds ZIP metadata - bytes may differ even for same weights. "
    "On PyTorch 2.x this xfail may show as XPASS because the timestamp is hardcoded.",
    strict=False,
)
def test_q2_naive_torch_save_byte_stability():
    # First attempt: seed the model, save twice with torch.save, compare hashes
    sd = train(torch_seed=42)
    torch.save(sd, "/tmp/_q2a.pt")
    torch.save(sd, "/tmp/_q2b.pt")
    match = hash_file("/tmp/_q2a.pt") == hash_file("/tmp/_q2b.pt")
    os.remove("/tmp/_q2a.pt")
    os.remove("/tmp/_q2b.pt")
    # Expecting this to fail (hashes differ due to ZIP metadata).
    # If it passes (XPASS), torch has fixed this internally - document it.
    assert not match


def test_q2_deterministic_save_fixes_byte_stability():
    # Write raw tensor bytes directly - no zip, no metadata, no surprises
    sd1 = train(torch_seed=42)
    sd2 = train(torch_seed=42)
    save_deterministic(sd1, "/tmp/_q2c.pt")
    save_deterministic(sd2, "/tmp/_q2d.pt")
    match = hash_file("/tmp/_q2c.pt") == hash_file("/tmp/_q2d.pt")
    os.remove("/tmp/_q2c.pt")
    os.remove("/tmp/_q2d.pt")
    assert match


# ===========================================================================
# Question 3: set_seed() seeds three RNG families. Do all three matter?
#
# Naive assumption: seed everything to be safe.
# Discovery: numpy and python RNG don't touch nn.Linear init or SGD.
# Only the torch seed is load-bearing for a pure PyTorch loop.
#
# First try numpy seed only - fails.
# Then python seed only - fails.
# Then torch seed only - passes.
# Then all three - also passes, but now we know why.
# ===========================================================================


@pytest.mark.xfail(reason="numpy seed alone doesn't control torch weight initialisation")
def test_q3_attempt_numpy_seed_only():
    sd1 = train(numpy_seed=42)
    sd2 = train(numpy_seed=42)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


@pytest.mark.xfail(reason="python random seed alone doesn't control torch weight initialisation")
def test_q3_attempt_python_seed_only():
    sd1 = train(python_seed=42)
    sd2 = train(python_seed=42)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_q3_torch_seed_alone_is_sufficient():
    # torch seed is the only one that controls nn.Linear init and SGD
    sd1 = train(torch_seed=42)
    sd2 = train(torch_seed=42)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_q3_all_three_seeds_also_works():
    # set_seed() seeds all three - still correct, but numpy/python are redundant here.
    # They become relevant if the pipeline ever uses np.random or random directly.
    sd1 = train(torch_seed=42, numpy_seed=42, python_seed=42)
    sd2 = train(torch_seed=42, numpy_seed=42, python_seed=42)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_q3_nuance_varying_numpy_with_fixed_torch_changes_nothing():
    # Confirms numpy is inert: different numpy seeds, same torch seed, same result
    sd1 = train(torch_seed=42, numpy_seed=1)
    sd2 = train(torch_seed=42, numpy_seed=9999)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


# ===========================================================================
# Question 4: What about data ordering?
#
# Shuffling the dataset changes the order gradients are applied.
# Even with the same seed, a different order means different final weights.
# This is a separate entropy source from weight initialisation.
#
# Observation: shuffle with a fixed DataLoader seed is deterministic
# but produces different weights than no-shuffle. Not wrong, just different.
# ===========================================================================


def test_q4_shuffle_changes_weights():
    # Confirms data order is an entropy source independent of the model seed
    perm = torch.randperm(5, generator=torch.Generator().manual_seed(7))
    sd_normal = train(torch_seed=42)
    sd_shuffled = train(torch_seed=42, inputs=INPUTS[perm], targets=TARGETS[perm])
    assert not all(torch.equal(sd_normal[k], sd_shuffled[k]) for k in sd_normal)


def test_q4_seeded_shuffle_is_internally_reproducible():
    # A seeded shuffle gives the same order both runs - deterministic, just different
    perm = torch.randperm(5, generator=torch.Generator().manual_seed(7))
    sd1 = train(torch_seed=42, inputs=INPUTS[perm], targets=TARGETS[perm])
    sd2 = train(torch_seed=42, inputs=INPUTS[perm], targets=TARGETS[perm])
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


# ===========================================================================
# Question 5: Are the entropy sources independent?
#
# Can fixing one accidentally fix the other? This is a 2x2 matrix:
#
#                  | torch.save  | det_save
#   no torch seed  |  XFAIL      |  XFAIL   <- weights wrong, format may be wrong
#   with torch seed|  see note   |  PASS    <- target
#
# The point of these tests is to prove you need to fix both independently.
# ===========================================================================


@pytest.mark.xfail(reason="no seed + torch.save: weights non-deterministic AND format unreliable")
def test_q5_matrix_no_seed_torch_save():
    sd1 = train(torch_seed=None)
    sd2 = train(torch_seed=None)
    torch.save(sd1, "/tmp/_m1a.pt")
    torch.save(sd2, "/tmp/_m1b.pt")
    match = hash_file("/tmp/_m1a.pt") == hash_file("/tmp/_m1b.pt")
    os.remove("/tmp/_m1a.pt")
    os.remove("/tmp/_m1b.pt")
    assert match


@pytest.mark.xfail(reason="no seed + det_save: format is fine but weights still non-deterministic")
def test_q5_matrix_no_seed_det_save():
    # det_save fixes the format, but without a seed weights differ run to run
    sd1 = train(torch_seed=None)
    sd2 = train(torch_seed=None)
    save_deterministic(sd1, "/tmp/_m2a.pt")
    save_deterministic(sd2, "/tmp/_m2b.pt")
    match = hash_file("/tmp/_m2a.pt") == hash_file("/tmp/_m2b.pt")
    os.remove("/tmp/_m2a.pt")
    os.remove("/tmp/_m2b.pt")
    assert match


def test_q5_matrix_seed_only_weights_are_fixed():
    # Seed fixes the weights. Whether the file bytes also match depends on
    # PyTorch version (see test_q2). Either way the weights themselves are stable.
    sd1 = train(torch_seed=42)
    sd2 = train(torch_seed=42)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_q5_matrix_seed_and_det_save_full_reproducibility():
    # Both sources fixed: same seed + deterministic save = identical bytes end to end
    sd1 = train(torch_seed=42)
    sd2 = train(torch_seed=42)
    save_deterministic(sd1, "/tmp/_m3a.pt")
    save_deterministic(sd2, "/tmp/_m3b.pt")
    h1, h2 = hash_file("/tmp/_m3a.pt"), hash_file("/tmp/_m3b.pt")
    os.remove("/tmp/_m3a.pt")
    os.remove("/tmp/_m3b.pt")
    assert h1 == h2


# ===========================================================================
# Question 6: Edge cases in the save function
#
# Non-contiguous tensors (e.g. weight.T) - used to raise RuntimeError in old
# NumPy. Now handled silently. Want to confirm .contiguous() doesn't change
# the byte content, just normalises the memory layout.
# ===========================================================================


def test_q6_noncontiguous_tensor_bytes_unchanged_by_contiguous():
    # If these differed, save_deterministic() would write different bytes
    # depending on how tensors happened to be laid out in memory
    sd = train(torch_seed=42)
    wT = sd["fc.weight"].T
    assert wT.numpy().tobytes() == wT.contiguous().numpy().tobytes()


def test_q6_nondeterministic_mode_informational():
    # On CPU this doesn't matter - on CUDA some kernels are nondeterministic
    # regardless of seeding. Not asserting, just printing the result.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd1 = train(torch_seed=42, deterministic=False)
    sd2 = train(torch_seed=42, deterministic=False)
    match = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    print(f"\n  [on {device}] deterministic=False, weights match: {match}")


if __name__ == "__main__":
    import pytest

    # This automatically triggers pytest if you run the file directly via python
    sys.exit(pytest.main(["-v", __file__]))
