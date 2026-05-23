# OpenVerifiableLLM — Deterministic Training & Verification Infrastructure

A toolkit for training language models whose entire training process is reproducible and independently auditable. Given the same data, configuration, and a fixed hardware stack, this infrastructure produces bit-identical models, and any deviation (corruption, tampering, or an honest mistake) is cryptographically detectable.

This is the **infrastructure** repository: the generic training-and-verification toolkit. The models trained with it (e.g. the Wikipedia models) live in a separate repository and depend on a pinned version of this one.

> **Status:** active development under Google Summer of Code 2026 with [AOSSIE](https://aossie.org). Some components below are proven and merged; others are planned. Each section marks which.

## The problem

Open-weight models are reproducible in principle but not verifiable in practice. You can download the weights, but you cannot prove what data they were trained on, what configuration produced them, or whether they were modified after training. A model ships with a report, and you trust the report.

There is no cryptographic link between a set of weights and the process that produced them. Post-training modification (fine-tuning, data injection, weight edits) is, by default, undetectable from the artifact alone.

This project treats verification as a property of the training pipeline itself rather than a claim bolted on afterward.

## What "verifiable" means here, precisely

It is worth being exact, because the word is often used loosely.

**What this infrastructure proves.** Given a fixed dataset snapshot, a fixed configuration, and the same hardware/software stack, an independent party can reproduce the exact model (bit-identical weights) or detect that a published artifact deviates from what was claimed. The verification is exact (a hash match), not approximate.

**What it does not prove.** A passing verification confirms that a training segment is *reproducible and internally consistent*. It does not, on its own, prove that training was *honest*, because a determined adversary can construct a checkpoint chain backwards that passes every spot-check (see [Threat model](#threat-model) below). The infrastructure raises the cost of forgery substantially; it does not reduce it to zero. That stronger guarantee requires cryptographic proof-of-training (zkML), which is out of scope at this scale and noted as future work.

This honesty is deliberate. The system is designed around *falsifiability*: its job is to fail reliably when assumptions are violated, not merely to pass when everything is correct.

## Key technical findings

These come from controlled experiments in the [baseline repository](https://github.com/ryoari/Verifiable-LLM-Baseline) and the project's experiment suite. They are the empirical basis for the design.

### Computational determinism is achievable; representational determinism is the catch

When seeds, initialization, data order, and configuration are fixed, the training computation itself is numerically stable. On a fixed single-GPU stack, two independent runs produce bit-identical weights (verified: identical SHA-256, identical final loss to the last digit).

But identical weights do not automatically produce identical files. PyTorch's `.pt` format stores checkpoints as ZIP archives with embedded timestamps and pickle metadata, so the bytes on disk change on every save even when the parameters are identical. This breaks naive file-level hash verification.

**The fix:** verify at the *tensor* level, not the file level. Extract weights into a canonical representation, serialize with a byte-stable format ([safetensors](https://huggingface.co/docs/safetensors)), and hash the raw tensor data. This is the difference between a model that is reproducible and a model that is verifiable.

| Determinism type | Property | Status |
|---|---|---|
| Computational | same config → same weight values | Verified (single GPU, fixed stack) |
| Representational | same weight values → same bytes on disk | Broken with `.pt`, fixed with safetensors |

### Loss-curve verification is insufficient

Comparing training loss trajectories across runs is not a reliable audit on its own. Two scenarios from the falsifiability suite show why:

- **Post-training sabotage:** weights mutated *after* training completes. The loss trajectory replays perfectly (the mutation happened after the replayed window). Only the tensor hash catches it.
- **File-level corruption:** a small corruption produces loss differences around 1e-8, indistinguishable from floating-point noise. Loss check passes; hash check fails.

The conclusion: hash-based tensor verification is necessary, not optional. An audit must operate at the tensor-hash level, and trajectory comparison and hashing are *both* needed because each catches failures the other misses.

### Determinism flags vs. seeding

In the current nanoGPT-scale experiments, fixing the RNG seed was the dominant factor; the model reproduced bit-identically even with `torch.use_deterministic_algorithms` off, because the operations it uses (dense matmul, layernorm, attention, embeddings) are already deterministic by default. The flag is kept on regardless: it is a no-op on the current model but a guarantee against silent regression if the architecture later touches a non-deterministic-by-default kernel (scatter/gather with duplicate indices, atomic-accumulation paths, certain convolution backward passes). This assumption is re-tested at larger scale rather than assumed to carry over.

## Architecture

The pipeline cryptographically links every stage from raw data to final weights.

```
Dataset (pinned dump)                    ── Merkle root over ordered chunks
        │
        ▼
Tokenization (deterministic BPE/SP)      ── config hash, binary tokens.bin
        │
        ▼
Deterministic training loop              ── full RNG + optimizer state control
        │
        ▼
Verification layer                       ── tensor-level SHA-256, safetensors
        │
        ▼
Signed manifest + transparency log       ── Sigstore / Rekor (planned)
        │
        ▼
Evaluation (factual, bias)               ── hash-linked into the chain
```

Each stage records its inputs and outputs into a manifest, and the manifests chain into a single pipeline hash, so any link can be independently checked.

## The two core deliverables

The infrastructure centers on two programs with a shared, versioned contract.

### 1. Trainer / chain producer

```
train(D, P) → M, [M0, M1, ..., Mn], [D1, ..., Dn]
```

Takes data `D` and parameters `P`, produces the final model `M` and the full sequence of incremental snapshots plus the data split used to produce them. The chain begins at `M0` (the deterministically-seeded initial model, before any training) so that even the first segment is verifiable.

Each snapshot is a **full training-state boundary**, not just weights. To allow exact replay of any segment, a snapshot includes:

- model weights (safetensors)
- optimizer state (e.g. Adam moments)
- complete RNG state (Python, NumPy, torch CPU, torch CUDA)
- LR-schedule position and step count
- dataloader position

This completeness is what makes segment replay reproduce bit-identically; omitting any of it breaks the guarantee.

### 2. Segment verifier

```
verify(P, Mk, D_{k+1}, M_{k+1}) → pass / fail
```

Takes a boundary snapshot `Mk`, the next data chunk `D_{k+1}`, the configuration `P`, and the claimed next snapshot `M_{k+1}`. It replays that single segment and checks the result.

The default test is a **bit-exact hash match**, valid on the same hardware/software stack, which leaves no tolerance window for a forged or corrupted step to hide in. Cross-hardware verification (where floating-point non-associativity across GPU architectures makes bit-exactness impossible) is available as a separate, explicitly-labeled mode with a documented tolerance. The strong mode is the default; the tolerant mode is opt-in.

This is what makes verification cheap. An auditor verifies any single ~1%-of-training segment at ~1% of the cost, samples a few at random, and gains high confidence without retraining the whole model.

## Falsifiability suite

A model is only as trustworthy as the test that tries to break it. The suite (ported into this repo as a `pytest` harness) provides adversarial coverage. A clean run must pass; every tampered run must fail.

| Scenario | What it tests | How it's caught |
|---|---|---|
| Clean audit | end-to-end reproduction | hashes + trajectory match |
| Bad seed | wrong RNG initialization | trajectory diverges, hash mismatch |
| Gradient noise | mid-training perturbation | trajectory diverges, hash mismatch |
| Post-training sabotage | weights edited after training | trajectory passes, **hash catches it** |
| Broken seal | ~1e-8 file corruption | trajectory passes, **hash catches it** |
| Prover / auditor split | two-party independent replay | segment replays bit-identically |

The last two scenarios are the point: they are invisible to any loss-curve-only audit.

## Threat model

State this plainly so the guarantees are not overread.

- **Catches:** accidental corruption, drift, post-training weight edits, file-level tampering, configuration mismatch, dataset substitution (the data Merkle root won't match).
- **Raises the cost of, but does not cryptographically prevent:** a determined forger constructing a checkpoint chain backwards to pass spot-checks. This is a known limitation of checkpoint-replay verification (see Fang et al. 2023, ["Proof-of-Learning Is Currently More Broken Than You Think"](https://arxiv.org/abs/2208.03567), which rebuts the original Proof-of-Learning construction in [Jia et al. 2021](https://arxiv.org/abs/2103.05633)).
- **Mitigation in scope:** publishing the ordered-dataset Merkle root and a transparency-log timestamp *before* training pins the inputs, so a forger cannot freely choose the data, which substantially raises the forgery bar.
- **Out of scope:** cryptographic proof of an honest gradient step (zkML). zkML can prove small-model *inference* but not *training* at meaningful scale as of 2026. Noted as future work, not promised.

### Supply-chain posture

The verification secures the model artifact, but the verifier and the training infra are themselves code that people download and run, which is exactly the layer recent supply-chain attacks target. Posture, not a separate workstream:

- dependencies are pinned and hash-locked (`uv` lockfile), which doubles as a defense against malicious mid-stream package updates
- infra/verifier releases are signed (Sigstore), so an auditor can confirm the verifier they run is the one actually published
- the small, separately-versioned infra repo has a deliberately minimal attack surface

## Hardware and reproducibility boundary

- **Bit-exact reproducibility is guaranteed on an identical hardware/software stack** (same GPU architecture, CUDA, cuDNN, PyTorch versions). The environment is pinned via the lockfile and recorded in the manifest.
- **Cross-hardware** (e.g. A100 vs H100), floating-point non-associativity means bit-exactness does not hold. This is measured and documented, not hidden, and is the use case for the verifier's tolerant mode.
- **Single GPU** is the supported, proven domain. Multi-GPU determinism is harder: the cross-device gradient all-reduce (NCCL) introduces a reduction whose order is not fixed by default. It is controllable for data-parallel training in fp32 with a pinned NCCL algorithm, at a throughput cost, and is treated as a measured experiment rather than an assumption. Tensor/pipeline parallelism is not in scope.

## Tech stack

Python, PyTorch, safetensors, NumPy, CUDA, SHA-256, Merkle trees, `uv`, `ruff`, `pytest`, GitHub Actions, Sigstore (planned), bitsandbytes, lm-evaluation-harness.

## Repository relationship

| Repo | Contains | Cadence |
|---|---|---|
| **This (infra)** | trainer, verifier, manifest schema, falsifiability suite, signing | code releases (semver) |
| **Models** | pinned dump pointers, training configs, published chains, manifests, eval reports | per training run |

The model repo pins an exact version of this infra repo, because a manifest is only meaningful against the exact infra version that produced it. Verification logic never lives in the model repo; it only produces and consumes manifests.

## Getting started

> Setup instructions are stabilizing as the core lands in the main repository. The intended flow:

```bash
# install (pinned, hash-locked dependencies)
uv sync

# run the falsifiability suite — clean passes, tampered fails
pytest tests/falsifiability

# train with a chain of snapshots
python -m ovllm.train --data <dump> --config <config> --out <dir>

# verify a single segment
python -m ovllm.verify --params <config> --from Mk --data D_{k+1} --expect M_{k+1}
```

## Acknowledgments

Developed for AOSSIE under GSoC 2026, with mentorship from the OpenVerifiableLLM team. The dataset, Merkle, manifest, and evaluation foundations were built by the wider project; this repository extends them with the deterministic, verifiable *training* layer.

## References

- Jia et al., *Proof-of-Learning: Definitions and Practice* (2021) — [arXiv:2103.05633](https://arxiv.org/abs/2103.05633)
- Fang et al., *"Proof-of-Learning" Is Currently More Broken Than You Think* (EuroS&P 2023) — [arXiv:2208.03567](https://arxiv.org/abs/2208.03567)
- safetensors format — [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)
- Sigstore model transparency — [github.com/sigstore/model-transparency](https://github.com/sigstore/model-transparency)
- Baseline prototype — [github.com/ryoari/Verifiable-LLM-Baseline](https://github.com/ryoari/Verifiable-LLM-Baseline)
