# OpenVerifiableLLM

**Deterministic training and independent verification for language models.**

OpenVerifiableLLM is an [AOSSIE](https://aossie.org) project building a training pipeline whose entire process is reproducible and independently auditable. Given the same data, configuration, and a fixed hardware stack, the pipeline produces bit-identical models, and any deviation (corruption, tampering, or an honest mistake) is cryptographically detectable.

The goal is not just to publish a model, but to publish a model whose training process can be verified rather than trusted.

---

## The reproducibility matrix (this build)

Earlier versions hashed a checkpoint and called it verified. The sharper claim this
build is organised around: **training reproducibility is universally assumed and
rarely verified — it breaks silently, and this tool surfaces the exact conditions
under which it fails.** The hook is the *failure*, not the success.

A single parametrized runner sweeps a grid of models × conditions and emits one JSON
record per cell, deliberately mixing reproducible and broken outcomes:

| | fp32 det-on | fp32 det-off | tf32 | bf16 | cross-GPU |
|---|---|---|---|---|---|
| **mlp** (control) | PASS | (verify on GPU) | bits≠fp32 (Ampere) | bits≠fp32 | (verify) |
| **gpt10m** (attention) | PASS | run-to-run FAIL on GPU | bits≠fp32 | bits≠fp32 | DIFF |
| **lstm** (cuDNN recurrent) | PASS | FAIL on GPU | bits≠fp32 | bits≠fp32 | DIFF |

Two comparisons are kept deliberately separate, because conflating them is the most
common reproducibility error:

- **Run-to-run** (`reproducible`, `first_divergence_step`): train the *same* config
  twice on the *same* hardware — identical bits? Broken by nondeterministic kernels
  (determinism OFF) and by different GPUs.
- **Agreement with the fp32 reference** (`vs_fp32`): does tf32/bf16 produce the same
  bits — or merely the same loss to a tolerance — as fp32? TF32/bf16 are perfectly
  run-to-run reproducible yet silently disagree with fp32.

**The verification bar (resolved).** `verify()` compares losses within `rel_tol=1e-6`
as a *diagnostic* (it localizes the first divergent step), but the verdict is the
exact parameter hash: a run that matches every loss to 1e-6 and differs by one bit
fails. Tolerance-based acceptance is precisely the loophole that broke
proof-of-learning ([Fang et al. 2023](https://arxiv.org/abs/2208.03567)) — forged
transitions hide inside the tolerance window, and bit-exact replay closes it. The
cost of that choice is hardware-boundedness: tf32/bf16 runs are run-to-run
reproducible yet bitwise disagree with fp32, so the bitwise bar is only defined
within a pinned hardware/software stack — which is exactly what the matrix measures.

**Security upgrade.** Checkpoints are now ed25519-signed and the signature is verified
*before* deserialization (`src/signing.py`). The previous path,
`torch.load(weights_only=False)` followed by a SHA-256 check, executes arbitrary code
at unpickle time — before the integrity check ever runs. Signing also makes the
"cryptographically signed" claim true (a SHA-256 is a checksum, not a signature).

### Quickstart

```bash
pip install -r requirements.txt
python demo.py                 # full arc on CPU (smoke config); --full on a GPU pod
python sweep.py --quick        # the matrix, tiny CPU preset
cd src && python reproducibility.py   # segmented-replay audit (CLEAN AUDIT PASS + scenarios)

# k-of-N chain audit: train a signed N-segment chain, then spot-check it
cd src && python chain.py train --model mlp --segments 4 --segment-steps 3 \
    --batch-size 4 --block-size 48 && python chain.py audit --k 2
```

See **RUNBOOK.md** for the exact pod commands that populate the GPU-only cells.

---

## Why this project exists

Open-weight models are reproducible in principle but not verifiable in practice. You can download the weights, but you cannot prove what data they were trained on, what configuration produced them, or whether they were modified after release. A model ships with a report, and the report has to be trusted.

There is no cryptographic link between a set of weights and the process that produced them, which makes post-training modification (fine-tuning, data injection, weight edits) effectively undetectable from the artifact alone.

OpenVerifiableLLM treats verification as a property of the training pipeline itself rather than something added afterward.

## What "verifiable" means here

The term is used precisely in this project.

**What the system proves.** Given a fixed dataset snapshot, a fixed configuration, and the same hardware/software stack, an independent party can reproduce the exact model (bit-identical weights) or detect that a published artifact deviates from what was claimed. Verification is exact (a hash match), not approximate.

**What it does not prove.** A passing verification confirms that a training segment is reproducible and internally consistent. It does not, on its own, prove that training was honest, because a determined adversary can construct a checkpoint chain that passes spot-checks (see [Threat model](#threat-model)). The system substantially raises the cost of forgery; it does not reduce it to zero. The stronger guarantee requires cryptographic proof-of-training (zkML), which is not tractable at this scale and is treated as future work.

This honesty is by design. The system is built around *falsifiability*: it must fail reliably when assumptions are violated, not merely pass when everything is correct.

## How it works

The pipeline cryptographically links every stage from raw data to final weights.

```
Dataset (pinned dump)              -- Merkle root over ordered chunks
        |
        v
Tokenization (deterministic)       -- config hash, binary tokens
        |
        v
Deterministic training loop        -- full RNG + optimizer state control
        |
        v
Verification layer                 -- tensor-level SHA-256, safetensors
        |
        v
Signed manifest + transparency log -- Sigstore / Rekor
        |
        v
Evaluation (factual, bias)         -- hash-linked into the chain
```

Each stage records its inputs and outputs into a manifest, and the manifests chain into a single pipeline hash so any link can be checked independently.

### Two core programs

**Trainer / chain producer.** Takes data and parameters, produces the final model along with a sequence of incremental snapshots and the data split used to produce them. The chain begins at the deterministically-seeded initial model (before any training) so that even the first segment is verifiable. Each snapshot is a complete training-state boundary (weights, optimizer state, full RNG state, schedule position, dataloader position), not just weights, because exact segment replay depends on restoring all of it.

**Segment verifier.** Takes a boundary snapshot, the next data chunk, the configuration, and the claimed next snapshot, then replays that single segment and checks the result. The default test is a bit-exact hash match (valid on the same hardware stack, with no tolerance window for a forged or corrupted step to hide in). Cross-hardware verification is available as a separate, explicitly-labeled mode with a documented tolerance.

This is what makes verification affordable: an auditor can verify any single segment at a small fraction of the full training cost, sample several at random, and gain high confidence without retraining the whole model.

## Design findings

These observations from the project's controlled experiments inform the architecture.

**Computational determinism is achievable; representational determinism is the catch.** With seeds, initialization, data order, and configuration fixed, training computation is numerically stable, and two independent runs on a fixed single-GPU stack produce bit-identical weights. However, identical weights do not produce identical files: PyTorch's `.pt` format embeds timestamps and pickle metadata, so the bytes change on every save. Verification therefore operates at the tensor level using a byte-stable format ([safetensors](https://huggingface.co/docs/safetensors)), not at the file level.

| Determinism type | Property | Status |
|---|---|---|
| Computational | same config produces same weight values | achievable (single GPU, fixed stack) |
| Representational | same weight values produce same bytes on disk | broken with `.pt`, resolved with safetensors |

**Loss-curve verification is insufficient on its own.** Trajectory comparison misses two important attacks: weights mutated after training completes (the replay window passes, only the hash catches it), and small file corruptions producing loss differences around 1e-8 that are indistinguishable from floating-point noise. Tensor-hash verification is necessary, and trajectory comparison and hashing are both used because each catches failures the other misses.

## Falsifiability suite

A clean run must pass; every tampered run must fail.

| Scenario | What it tests | How it's caught |
|---|---|---|
| Clean audit | end-to-end reproduction | hashes + trajectory match |
| Bad seed | wrong RNG initialization | trajectory diverges, hash mismatch |
| Gradient noise | mid-training perturbation | trajectory diverges, hash mismatch |
| Post-training sabotage | weights edited after training | trajectory passes, hash catches it |
| Broken seal | ~1e-8 file corruption | trajectory passes, hash catches it |
| Prover / auditor split | two-party independent replay | segment replays bit-identically |

## Threat model

Stated plainly so the guarantees are not overread.

- **Catches:** accidental corruption, drift, post-training weight edits, file-level tampering, configuration mismatch, and dataset substitution (the data Merkle root will not match).
- **Raises the cost of, but does not cryptographically prevent:** a determined forger constructing a checkpoint chain that passes spot-checks. This is a known limitation of checkpoint-replay verification (see Fang et al. 2023, ["Proof-of-Learning Is Currently More Broken Than You Think"](https://arxiv.org/abs/2208.03567), rebutting [Jia et al. 2021](https://arxiv.org/abs/2103.05633)) — but see the audit arithmetic below for what determinism changes about it.
- **Mitigation:** publishing the ordered-dataset Merkle root and a transparency-log timestamp before training pins the inputs, so a forger cannot freely choose the data, which raises the forgery bar.
- **Out of scope:** cryptographic proof of an honest gradient step (zkML), which can prove small-model inference but not training at meaningful scale today.

### What a spot-check audit buys

The soundness of segment-replay verification can be stated precisely rather than
qualitatively. Setup: training is published as a chain of **N** segments; every
input that determines the trajectory (ordered-dataset Merkle root, configuration,
seed, initialization) is committed to a transparency log *before* training; replay
is bit-exact on the pinned stack.

**Why determinism changes the game.** Every attack in Fang et al. forges checkpoint
transitions that are not real training steps but land *within the verifier's
tolerance window*. Under bit-exact replay the window is zero: with all inputs
committed, the entire trajectory is a deterministic function of the commitment, so
a published model that differs from that function must contain at least one segment
transition that does not replay — and there is no tolerance for it to hide in. A
segment either reproduces the exact hash or it fails. The adversary's remaining
moves are (a) choosing malicious inputs *before* commitment (data poisoning — a
real but separate problem), and (b) hoping the forged segment is never sampled.

**The sampling bound.** An auditor who replays *k* of the *N* segments, chosen
uniformly at random, catches a chain containing *b* invalid segments with
probability 1 − C(N−b, k) / C(N, k). For the minimal forgery (b = 1) this is
exactly **k/N**. With *m* independent auditors each sampling *k* segments and
publishing their results to the transparency log, detection probability is at
least 1 − (1 − k/N)^m. Stated honestly: a single-bad-segment forgery survives one
k-sample audit with probability 1 − k/N; confidence comes from audit *volume*,
which is why audit results belong in a public log.

**The cost model.** Auditor compute ≈ (k/N) × full training cost, plus downloading
k + 1 boundary checkpoints. Publisher storage ≈ N × (weights + optimizer state) —
for Adam, roughly 3× the weight size per checkpoint. Segment length is therefore a
tunable trade: shorter segments mean cheaper individual audits and finer forgery
localization but more storage; longer segments the reverse.

**Open question (research direction).** Whether a single forged transition — a
jump from a genuine trajectory onto a target model — is statistically detectable
*without* replaying it (step-norm outliers, update direction vs. plausible
gradients). A positive result would push effective soundness from k/N toward 1.

### Supply-chain posture

Verification secures the model artifact, but the verifier and training code are themselves software that people download and run. Accordingly: dependencies are pinned and hash-locked, releases of the verification tooling are signed (so an auditor can confirm the tool they run is the one published), and the verification infrastructure is kept small to minimize attack surface.

## Scope and boundaries

- **Bit-exact reproducibility is guaranteed on an identical hardware/software stack.** The environment is pinned and recorded in the manifest.
- **Cross-hardware** reproducibility (e.g. different GPU architectures) does not hold bit-exactly due to floating-point non-associativity; this is measured and documented, and is the use case for the verifier's tolerant mode.
- **Single GPU** is the supported, validated domain. Multi-GPU determinism is harder because the cross-device gradient all-reduce introduces a reduction whose order is not fixed by default; it is controllable for data-parallel training under specific conditions and is treated as a measured experiment rather than an assumption. Tensor and pipeline parallelism are out of scope.

## Repository structure

OpenVerifiableLLM is organized as two repositories:

| Repository | Contains |
|---|---|
| **Infrastructure** | trainer, verifier, manifest schema, falsifiability suite, signing tooling |
| **Models** | pinned dataset pointers, training configs, published checkpoint chains, manifests, evaluation reports |

A model repository pins an exact version of the infrastructure, because a manifest is only meaningful against the exact version that produced it. Verification logic lives only in the infrastructure; model repositories produce and consume manifests but do not reimplement verification.

## Tech stack

Python, PyTorch, safetensors, NumPy, CUDA, SHA-256, Merkle trees, `uv`, `ruff`, `pytest`, GitHub Actions, Sigstore, bitsandbytes, lm-evaluation-harness.

## Getting started

> Setup instructions are stabilizing as the core lands. The intended flow:

```bash
# install pinned, hash-locked dependencies
uv sync

# run the falsifiability suite (clean passes, tampered fails)
pytest tests/falsifiability

# train, producing a chain of verifiable snapshots
python -m openverifiablellm.train --data <dump> --config <config> --out <dir>

# verify a single segment
python -m openverifiablellm.verify --params <config> --from <Mk> --data <chunk> --expect <Mk+1>
```

## Contributing

Contributions are welcome. The project favors a research-oriented, assumption-first approach: validate that an abstraction holds before building on top of it, and design features to be falsifiable.

- Discussion happens in the [AOSSIE Discord](https://aossie.org); keep technical decisions public.
- Open an issue before substantial work so scope can be aligned with maintainers.
- Run `ruff` and the test suite before submitting; the determinism checks in CI are required to pass.
- Good first issues are labeled in the issue tracker.

See `CONTRIBUTING.md` for details.

## License

See [`LICENSE`](LICENSE).

## References

- Jia et al., *Proof-of-Learning: Definitions and Practice* (2021) -- [arXiv:2103.05633](https://arxiv.org/abs/2103.05633)
- Fang et al., *"Proof-of-Learning" Is Currently More Broken Than You Think* (EuroS&P 2023) -- [arXiv:2208.03567](https://arxiv.org/abs/2208.03567)
- safetensors format -- [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)
- Sigstore model transparency -- [github.com/sigstore/model-transparency](https://github.com/sigstore/model-transparency)
