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

**Measured, not asserted** (evidence in `proofs/chain_a40/`, commands in RUNBOOK §8):
a 116M-param model trained 20 × 500 steps on enwik8 on an A40 in strict
deterministic fp32 produced a chain whose k=3 sampled audit replayed bit-exactly,
at a measured cost ratio of 0.173 against the theoretical k/N = 0.15 (the gap is
per-replay setup: checkpoint load + hashing). Chain storage measured 1.36 GB per
boundary, matching the 3×-weights prediction for Adam.

**The hardware equivalence class, measured** (evidence in `proofs/chain_cross/`):
the same 116M-param chain (trained on an RTX A4000) was audited from three
positions with the identical sampled segments —

| Auditor | Relation to prover | Verdict |
|---|---|---|
| same pod | same physical GPU | **PASS** (bit-exact) |
| second RTX A4000 pod | same GPU model, different physical GPU | **PASS** (bit-exact) |
| L4 pod, different datacenter | different architecture (Ampere → Ada) | **FAIL** (closing hash mismatch, both segments) |

Two load-bearing details: verification is portable across *instances* of the same
GPU model — an auditor needs the prover's GPU model and software stack, not the
prover's machine — and in the cross-architecture failure the *opening* boundary
hashes matched while the *closing* hashes diverged, isolating the divergence to
replay arithmetic rather than transfer corruption. The auditors fetched the chain
over plain untrusted HTTP; the ed25519 signatures over the manifest and every
boundary are what established integrity.

**Replay-free forgery detection (first results).** Because boundaries are full
training states, a forged transition must be consistent not just with the weights
but with the Adam moments it carries — and that consistency is checkable in
O(params) without any replay (`src/forgery.py`: Adam-reachability of the weight
delta, robust delta-norm profile, moment-direction consistency). Flagged segments
get replayed first, pushing effective detection above the k/N floor. Measured at
116M params on an L40S (evidence in `proofs/forgery_l40s/`, zero false positives
in every condition):

| Forgery | S=200 | S=100 | S=10 |
|---|---|---|---|
| alt-seed splice (classic PoL spoof) | detected+localized | detected+localized | detected+localized |
| interpolation toward target, α=0.10 | detected+localized | detected+localized | detected+localized |
| interpolation toward target, α=0.03 | missed | detected+localized | detected+localized |
| gaussian weight edit, σ=1e-2 rel. | missed | missed | missed |

Two findings. First, **segment length is a detectability dial, not just a
storage/audit-cost dial**: the stealthy 3% splice is invisible at 200-step
segments and cleanly flagged at 100 and below — shorter segments tighten the
reachability envelope a fixed-size forgery must hide inside. Second, honest
limits: small gaussian edits evade these detectors at scale (they are also
near-useless to an attacker, and remain caught by sampled replay and — for the
final model — by the manifest hash directly). The sound guarantee remains the
replay.

**The arms race (smart forgers who forge the moments too).** Three escalations
were tested at the same scale and segment lengths (evidence in
`proofs/smartforger_l40s/`, zero false positives throughout): *sf-aligned* sets
moments perfectly anti-aligned with its fake delta — it defeats a one-sided
moment check, so the envelope is two-sided (too consistent is as damning as too
little), and it is flagged at every segment length. *sf-calibrated* engineers
its moment cosine into the genuine envelope and copies the genuine second
moments — its moment forging *works* (no moment flag at S=200/100), but the
weight-side detectors catch the splice anyway, and at S=10 the moment-continuity
envelope flags it too. *sf-freshv* fabricates second moments from scratch and
trips a **hard invariant**: Adam's v-update adds a non-negative term each step,
so `v_{k+1} ≥ β₂^S·v_k` elementwise for *any* genuine trajectory (β₂ = 0.999
keeps 82% of v across even 200 steps) — a necessary condition, not a heuristic,
that fabricated moments must satisfy at every one of ~10⁸ coordinates. The clean
separation: **moment forging only buys the attacker anything when the weight
delta is also small, and small-delta forgeries at long segments are exactly the
residual hole that sampled replay and shorter segments close.**

### Supply-chain posture

Verification secures the model artifact, but the verifier and training code are themselves software that people download and run. Accordingly: dependencies are pinned and hash-locked, releases of the verification tooling are signed (so an auditor can confirm the tool they run is the one published), and the verification infrastructure is kept small to minimize attack surface.

## Scope and boundaries

- **Bit-exact reproducibility is guaranteed on an identical hardware/software stack.** The environment is pinned and recorded in the manifest. Measured at 116M params: the equivalence class is the *GPU model plus software stack*, not the physical machine — a chain trained on one RTX A4000 verified bit-exactly on a different RTX A4000 in a different pod (`proofs/chain_cross/`).
- **Cross-hardware** reproducibility (different GPU architectures) does not hold bit-exactly due to floating-point non-associativity; measured at 116M params (Ampere-trained chain audited on Ada: every replayed segment fails on the closing hash). This is the use case for the verifier's tolerant mode.
- **Single GPU** is the supported, validated domain. Multi-GPU determinism is harder because the cross-device gradient all-reduce introduces a reduction whose order is not fixed by default; it is controllable for data-parallel training under specific conditions and is treated as a measured experiment rather than an assumption (`src/ddp_repro.py` is the ready-to-run probe). Tensor and pipeline parallelism are out of scope.

### The execution-variant envelope (measured)

Real training uses fused attention and compilation. Measured on an L40S
(gpt10m, 300 steps, dropout 0, twin runs per cell; evidence in
`proofs/envelope_l40s/`):

| Variant | Run-to-run (det on) | Run-to-run (det off) | Bits vs eager/manual |
|---|---|---|---|
| eager manual attention | reproducible | reproducible | reference |
| SDPA math backend | reproducible | reproducible | **differ** |
| SDPA efficient backend | reproducible | reproducible | **differ** |
| SDPA flash backend | *no fp32 kernel* | *no fp32 kernel* | n/a |
| torch.compile (manual) | reproducible | **NOT reproducible** | **differ** |

Four consequences. (1) **Every execution variant is its own bit-universe**: even
the SDPA *math* backend, nominally the same arithmetic, produces different bits
than manual attention — so the execution variant (`attn_impl`, `sdpa_backend`,
`compile`) is part of the committed config, and the auditor must replay with the
producer's exact variant (the chain format already enforces this: the variant
rides in the signed manifest). (2) **torch.compile is compatible with
verification**: compiled training is bit-exact run-to-run under strict
determinism, and a checkpoint chain trained with `compile: true` in its
commitment **passes the full segment audit bit-exactly**
(`proofs/envelope_l40s/compiled_chain_audit.json`). Compile without determinism
flags breaks run-to-run reproducibility — the flags are load-bearing, not
ceremonial. (3) **FlashAttention is outside the fp32 envelope entirely** (no
fp32 kernel exists); flash-based training lives in bf16/fp16, where bitwise
agreement with an fp32 reference is already off the table — extending the
verified envelope there is open work. (4) Deterministic-mode kernel *refusals*
are themselves audit-relevant data: a variant that strict mode rejects cannot
promise a replayable trajectory.

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
