# OpenVerifiableLLM End-to-End Evaluation

Reviewer: dhruvi-16-me  ·  Date: 2026-10-28  ·  Commit reviewed: `748280a`

I read the docs and code, downloaded the release (no checkpoints) on macOS with Python 3.12 and torch 2.14.1
(CPU), ran the CPU check with `ovl.py` from v1.0.0 and the timestamp checks, ran the tests and tried tampered
copies of the release. `simple/ovl.py` is identical at v1.0.0 and `748280a`. Skipped GPU check.

## 1. What was tried

| Area | Flows | Done? |
|---|---|---|
| First read | README, top to bottom | ✅ |
| Run the model | Ollama and/or `ovl.py chat` | skipped |
| Check the weights | Download, `sha256sum -c`, `ovl.py verify --weights` | ✅ `conversation-model.pt` MATCH entry 37; `base-model.pt` FAILs (H1) |
| Check the record | `ovl.py verify --anchor`, `openssl ts -verify` | ✅ chain head MATCH (38 entries); both timestamps OK once the CA path is fixed (L1) |
| Try to break it | Tampered files | ✅ weights +1e-6, edited last chain entry, truncated chain, missing `anchor.json`, `--anchor` to a non-anchor file, plus edits to `anchor.json` and the chain checked against the timestamps |
| Recompute a segment | One-click RunPod check (optional) | skipped GPU check |
| `ovllm verify` | Normal and tampered model directory | ⚠️ read `src/verifier.py` path handling only |
| Tests | `pytest tests/simple` | ✅ 29 passed, 1 skipped, after an undocumented install step (M4) |
| Contributing | CONTRIBUTING.md walk-through | ⚠️ test setup only (M4, L6) |

## 2. Summary

The code does what it says on its own terms: a broken hash link, a mismatched anchor or changed weights all fail.
The weak points are in what the docs lead a reader to trust. The CPU check prints MATCH without anything outside
the download being checked, a missing `anchor.json` is skipped without a word, and the key terms are used before
they're explained. One check gives a wrong answer on a released file: the genuine base model fails `verify`.

| Severity | Count |
|---|---|
| High | 1 |
| Medium | 4 |
| Low | 7 |
| Reported privately | 1 |

## 3. High

**H1: the released `base-model.pt` fails `verify`.**

- Steps: in the downloaded `release/simple-run`, run
  `OVL_DEVICE=cpu python ovl-code/simple/ovl.py verify --run . --weights base-model.pt`
- Actual: `weights base-model.pt: chain entry 32 (transition), not a released model (expected base or final)`,
  `verify FAIL`, exit 1. The file passes `sha256sum -c` against `release-hashes.txt`.
- Expected: `MATCH chain entry 31`. Entry 31 is `base`, and entry 32 (`transition`) records the same model digest,
  because the phase change doesn't touch the weights. `verify` takes the last matching entry, so a `base` match is
  never reached.
- Where: `simple/ovl.py:544-545` (`hit[-1]`); accepting any `base`/`final` entry in `hit` would fix it.

1 security finding reported privately.

## 4. Medium

| ID | What you saw | Steps | Where |
|---|---|---|---|
| M1 | Without `--anchor`, a missing `anchor.json` is skipped with no output. The run can still end in `verify PASS`, so the user never learns the anchor wasn't checked. | Delete `anchor.json`, run `ovl.py verify --run . --weights conversation-model.pt` | `simple/ovl.py:526-534` (no `else` warning); README "Getting started" doesn't mention it |
| M2 | The CPU check stops at `verify`. It never runs the `openssl ts -verify` step, which is the only part tied to something outside the download. A reader who follows the README sees "chain head MATCH" and reasonably takes it as the full check. | Follow README → "Getting started" | README.md:140-156; `docs/guides/check_a_segment.md:77-83` |
| M3 | Jargon is used before it's defined. "Segment", "chain entry 37" and "boundary" appear in the image alt text and the bullet list (lines 28-48) before the explanation at line 62. "Anchor" and "chain head" are never defined in the README. | Read the README top to bottom | README.md:28-63 |
| M4 | CONTRIBUTING says the `simple/` tests need only "PyTorch plus the pinned packages". With exactly that, pytest stops before any test with `ModuleNotFoundError: No module named 'ovl_pipeline'`, because the shared `tests/conftest.py` imports `scripts/rental_quote.py`. After `pip install -e .` they pass (29 passed, 1 skipped). | Fresh venv with the `simple/README.md` packages, then `OVL_DEVICE=cpu OVL_TIMESTAMP=0 python -m pytest tests/simple` | `CONTRIBUTING.md:65`, `tests/conftest.py:11`, `scripts/rental_quote.py:15` |

## 5. Low

| ID | What you saw | Where |
|---|---|---|
| L1 | The `openssl ts -verify` command uses a Linux-only CA path (`/etc/ssl/certs/ca-certificates.crt`), so on macOS it fails with `Error loading file /etc/ssl/certs/ca-certificates.crt` and no hint. `-CAfile /etc/ssl/cert.pem` works on macOS. | `docs/guides/check_a_segment.md:81` |
| L2 | The repo root is crowded: `Legacy/`, `project/`, `src/ovl_pipeline/`, `experiments/`, `notebooks/`, `proofs/` and loose `demo.py`, `sweep.py`, `run_experiment.py` all sit next to `simple/`. It isn't obvious that `simple/ovl.py` is the thing to look at. A short "where to look" table near the top of the README would help. | repo root |
| L3 | `verify` appends to `verifications.jsonl` inside `--run`, so it writes into the downloaded release folder (one CPU check grew it from 7355 to 7676 bytes). That's surprising for a read-only check. | `simple/ovl.py:560-561` |
| L4 | Entry 0 records a `recipe` hash, but `verify` never compares it with the running `RECIPE`. With the wrong `ovl.py` version, a user gets an unexplained segment MISMATCH (after ~20 GPU minutes) instead of an immediate "this code doesn't match the recipe the chain was trained with". | `simple/ovl.py:422` (written), `cmd_verify` (not checked) |
| L5 | The "prepared data differs" check hashes only each phase's `stream.json`, not the token files the loader reads. Changed tokens are still caught later as a segment MISMATCH, but the up-front data check covers less than its message implies. | `simple/ovl.py:536-537` |
| L6 | CONTRIBUTING's hash-locked setup was compiled for Python 3.12 (see the lock header), but neither CONTRIBUTING nor `pyproject.toml` (`>=3.10`) says so. On 3.10/3.11, `--require-hashes` installs may fail with no hint why. | `CONTRIBUTING.md:43-56`, `requirements/preparation.lock:2` |
| L7 | `--anchor` pointed at a JSON file that isn't an anchor (e.g. `recipe.json`) crashes with a `KeyError: 'chain_entries'` traceback instead of saying the file isn't an anchor. The run fails, so this is about the message only. | `simple/ovl.py:530` |
| L8 | The summary line counts only the weights check: a run that checked the anchor and the weights prints `verify PASS: 1 checks`, and the anchor result isn't recorded in `verifications.jsonl`. | `simple/ovl.py:535`, `simple/ovl.py:562` |

## 6. What worked well

- `read_chain` checks every `previous` link and index, so deleting, reordering or editing a middle entry fails
  immediately.
- `load_checkpoint` checks both the control block and the full model and optimizer digest before trusting a
  checkpoint.
- The weights check only accepts `base`/`final` entries, and the message explains when it matched something else
  (though see H1).
- All the documented break-it cases fail as they should: weights changed by 1e-6 (`NOT in the chain`), an edited
  last chain entry and a truncated chain (`chain head differs`).
- Both RFC 3161 timestamps on `anchor.json` verify, and both fail on any edited copy of `anchor.json`.
- `verify` with no checks results in FAIL, not an empty PASS.
- The README is upfront that the old `code/` copy predates the anchor check.
- The model is described honestly: README.md:93 says it "often gets facts wrong" and puts the method, not the
  model, forward. That sets the right expectations for a 23M model.
- Checkpoints and weights are loaded with `torch.load(..., weights_only=True)`, so a tampered `.pt` can't run code
  when it's loaded.
- CONTRIBUTING gives a per-area test command (`simple/`, `ovllm verify`, everything else), which makes a first
  change approachable.
