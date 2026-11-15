# Segment Data Slices — Feasibility Report

**Issue:** AOSSIE-Org/OpenVerifiableLLM #299
**Branch:** `feat/segment-data-slices`
**Scope:** Synthetic-data feasibility prototype only
**Status:** Initial extraction/reconstruction milestone completed

## 1. Objective

Evaluate whether a selected training segment can be represented as a compact saved package that reconstructs the same batches and continuation positions as the existing full-data loader.

This report records a feasibility experiment. It does not claim production-data compatibility or a completed cryptographic verification design.

## 2. Prototype approach

The prototype extracts the windows required by a selected segment and stores the information needed to reconstruct them, including:

- token values and target masks;
- the global `target_start` value;
- the predecessor token needed when a window begins inside a document;
- the document-row location and window start;
- the continuation position after each extracted window.

Packages are serialized as JSON under `experiments/segment_slices/saved_packages/`. The `build` and `verify` commands are run in separate Python processes. During `verify`, reconstruction uses the saved package rather than reading the original synthetic `documents.jsonl`, `tokens.u16`, or `mask.u8` files.

## 3. Functional results

Two synthetic segments were tested.

| Segment | Start position | End position | Extracted windows | Reconstructed batches | Package size |
|---|---:|---:|---:|---:|---:|
| 1 | `[61, 0]` | `[122, 4]` | 2 | 1 | 525 bytes |
| 2 | `[122, 4]` | `[183, 0]` | 1 | 1 | 389 bytes |

For both segments:

- reconstructed tensor contents matched the full-loader output;
- ending/continuation positions matched;
- saved packages were reloaded and checked in a separate process;
- stored batch digests and positions matched.

The second segment starts partway through a document. The prototype preserved the predecessor token and global target offset required to reproduce the full-loader batch.

## 4. Regression tests

Command:

```powershell
& $python -m pytest tests/simple/test_segment_slice_loader.py -q
```

Result:

```text
3 passed
```

The existing synthetic loader tests for reference/fast-loader equivalence and segment resume behavior passed.

## 5. Negative package tests

Temporary copies of the saved package were modified. The original saved package files were not changed.

| Mutation | Observed result |
|---|---|
| Change a token without changing the stored batch digest | Rejected: digest mismatch |
| Reorder windows | Rejected: continuation-position mismatch |
| Remove a window | Rejected: continuation-position mismatch |

All three negative tests passed: the verifier rejected each modified package.

## 6. Security limitation discovered

A separate controlled test changed a token from `20` to `21` and then recalculated the expected batch digest inside that same package.

The current verifier accepted the modified package.

This is expected because the package contains both the data and the expected digest used to check that data. An attacker who can rewrite the package can rewrite both. The current check therefore establishes **internal consistency**, not authenticity against a trusted source.

The current prototype does not provide a cryptographic proof that the included token and mask data are part of the preparation committed to by the existing OVL chain.

## 7. What has been demonstrated

The experiment supports these limited conclusions:

1. For the tested synthetic inputs, segment windows can be extracted and serialized.
2. The saved windows can reconstruct batches with matching tensor contents and continuation positions.
3. The tested mid-document resume case can preserve the predecessor token and global target offset.
4. The current verifier detects simple accidental corruption, reordering, and omission when the stored expected digest/positions are left unchanged.

## 8. What has not been demonstrated

The experiment does **not** yet demonstrate:

- compatibility with the real prepared training dataset;
- actual download-size reduction on production data;
- exact model-state or checkpoint replay equivalence;
- authentication of slice bytes against a trusted, chain-anchored commitment;
- compatibility with the existing production verification contract;
- resistance to an attacker who can modify both package contents and in-package expected digests.

The synthetic package sizes are not evidence of production download savings.

## 9. Next technical milestone

Before changing production verification, define and test how a segment package can prove that its selected token/mask byte ranges belong to the exact preparation already authenticated by OVL.

The design should identify:

1. which existing chain-authenticated metadata can anchor a prepared-data manifest;
2. how the manifest commits to file identity, byte length, chunking parameters, and content roots;
3. how a segment package carries range proofs for only the required chunks;
4. how the verifier checks those proofs against a trusted root obtained from the authenticated preparation metadata;
5. how the design rejects altered, reordered, omitted, duplicated, or substituted chunks.

The proof format must not silently replace or weaken the current OVL verification guarantees. Any proposed new commitment must be explicitly linked to the existing authenticated preparation metadata.

## 10. Reproduction commands

From the issue worktree, with `$python` pointing to the existing Python 3.12 virtual-environment interpreter:

```powershell
& $python ".\experiments\segment_slices\prototype_saved_slice.py" build
& $python ".\experiments\segment_slices\prototype_saved_slice.py" verify
& $python -m pytest tests/simple/test_segment_slice_loader.py -q
& $python ".\experiments\segment_slices\test_saved_package_tampering.py"
& $python ".\experiments\segment_slices\test_package_self_authentication.py"
```

The last command is expected to demonstrate the current self-authentication limitation; acceptance of the intentionally modified package is not a successful security guarantee.

## 11. Repository validation

- Full test suite: 118 passed, 9 subtests passed (after rebasing on upstream/main).
- Ruff: All checks passed.
- Segment-loader tests: 3 passed.
- Saved-package build and verification: passed.
- Package tampering checks: all three cases rejected.
- Self-authentication limitation: reproduced as expected.
---

**Conclusion:** The synthetic feasibility prototype demonstrates exact batch/position reconstruction for the tested segments, including a mid-document start. It also exposes the central unresolved requirement: slice contents must be authenticated against a trusted commitment outside the slice package.
