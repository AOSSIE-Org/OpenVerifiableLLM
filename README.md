# OpenVerifiableLLM

**Deterministic training and independent verification for language models.**

OpenVerifiableLLM is an [AOSSIE](https://aossie.org) project building a training pipeline whose entire process is reproducible and independently auditable. Given the same data, configuration, and a fixed hardware stack, the pipeline produces bit-identical models, and any deviation (corruption, tampering, or an honest mistake) is cryptographically detectable.

The goal is not just to publish a model, but to publish a model whose training process can be verified rather than trusted.

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
- **Raises the cost of, but does not cryptographically prevent:** a determined forger constructing a checkpoint chain that passes spot-checks. This is a known limitation of checkpoint-replay verification (see Fang et al. 2023, ["Proof-of-Learning Is Currently More Broken Than You Think"](https://arxiv.org/abs/2208.03567), rebutting [Jia et al. 2021](https://arxiv.org/abs/2103.05633)).
- **Mitigation:** publishing the ordered-dataset Merkle root and a transparency-log timestamp before training pins the inputs, so a forger cannot freely choose the data, which raises the forgery bar.
- **Out of scope:** cryptographic proof of an honest gradient step (zkML), which can prove small-model inference but not training at meaningful scale today.

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
