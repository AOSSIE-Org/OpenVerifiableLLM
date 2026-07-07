# Determinism Repairs Proof-of-Learning: Verifiable Training via Bit-Exact Segment Audits

*Working draft — target: ML-security workshop (SaTML / satellite). All numbers
are reproduced by artifacts in `proofs/` of this repository; commands in
RUNBOOK.md §8–9.*

## Abstract

Proof-of-learning (PoL) was proposed to let a third party verify that published
model weights resulted from a claimed training run, and was subsequently broken:
every known attack forges checkpoint transitions that land inside the verifier's
*tolerance window*, which exists because training is assumed nondeterministic.
We show that removing the tolerance repairs the scheme. With bit-exact
deterministic training and all trajectory-determining inputs (data Merkle root,
configuration, seed, initialization, execution variant) committed before
training, the entire trajectory is a deterministic function of the commitment: a
segment either replays to the exact parameter hash or it fails, and a k-of-N
random segment audit catches a minimal forgery with probability exactly k/N at
(k/N)× training cost. We characterize what this buys and what it costs on real
hardware at 116M parameters: (i) sampled audits replay bit-exactly at a measured
cost ratio of 0.173 versus the theoretical 0.15; (ii) the hardware equivalence
class is the *GPU model plus software stack*, not the physical machine — an
auditor on a different physical GPU of the same model verifies bit-exactly,
while a different architecture fails legibly on closing hashes only; (iii)
because chain boundaries are full optimizer states, cheap replay-free statistics
detect forged transitions before any replay, and *segment length* emerges as a
detectability dial: a stealthy 3%-interpolation forgery is invisible at 200-step
segments and reliably localized at 100 and below, at zero false positives —
robust to attackers who forge the optimizer moments too, one of whose evasions
is excluded by a hard invariant of Adam (v_{k+1} ≥ β₂^S·v_k elementwise); and
(iv) torch.compile is *compatible* with verification — a chain trained under a
committed `compile: true` passes the full bit-exact audit — while FlashAttention
is structurally outside the fp32 envelope. We state the residual gaps plainly:
small-delta forgeries at long segments evade replay-free detection, data
poisoning before commitment is out of scope, and multi-GPU all-reduce ordering
remains an open measurement.

## 1. Introduction

Open-weight models are reproducible in principle and unverifiable in practice:
weights ship with a report, and the report must be trusted. Jia et al. [1]
proposed proof-of-learning — publish intermediate checkpoints; a verifier
replays spot-checked segments and accepts if the replay lands *near* the claimed
checkpoint. Fang et al. [2] then showed the scheme is "more broken than you
think": adversaries can synthesize checkpoint transitions that were never
gradient steps yet pass the approximate check. The vulnerability is the
tolerance window, and the window exists only because training is assumed
irreproducible.

This paper's premise is that the assumption is an engineering choice, not a law.
On a pinned single-GPU stack with deterministic kernels, training is bit-exact
run-to-run. We rebuild PoL on that foundation and measure what it costs.

**Contributions.**
1. A verification protocol (Section 3) in which checkpoint-chain boundaries are
   *full training states* (weights, Adam moments, all four RNG streams) and
   every trajectory-determining input — including the execution variant — is
   committed before training. Zero-tolerance replay makes segment verification
   sound; we give the exact sampling bound and cost model.
2. An empirical characterization of the *hardware equivalence class* within
   which bit-exact verification is defined (Section 5): same GPU model on a
   different physical machine verifies; a different architecture fails, and
   fails legibly.
3. Replay-free forgery detectors over full-state boundaries (Section 6),
   including a hard invariant of Adam's second moment, an adversarial
   evaluation against moment-forging attackers, and the finding that segment
   length is a third axis of the checkpoint-spacing trade: storage vs. audit
   cost vs. free detection power.
4. The execution-variant envelope (Section 7): fused-attention backends and
   compilation each define their own "bit-universe," compilation is verifiable
   when committed, and FlashAttention cannot participate in the fp32 recipe at
   all.

## 2. Threat model

A **prover** trains a model and publishes: the commitment C = (ordered-dataset
Merkle root, full configuration including execution variant, seed,
initialization rule), timestamped in a transparency log *before* training; a
chain of N signed boundary checkpoints B₀…B_N (B₀ is the seeded init); and a
signed chain manifest binding per-boundary parameter hashes to C.

An **auditor** holds C, the chain, and hardware from the same equivalence class.
They may replay any segment i: restore B_i in full, run the committed S steps,
and compare the resulting parameter hash to B_{i+1} exactly.

The **adversary** is the prover: they want to publish a final model
M′ ≠ T(C) — e.g., a backdoored model — with a chain that passes audits. They
control everything after commitment. They do not control C (it is committed and
logged), and cannot forge ed25519 signatures. Data poisoning *within* C — 
choosing malicious-but-committed data — is real and out of scope here; it is a
data-auditing problem, not a replay problem. Integrity of transport is not
assumed: in our experiments auditors fetch chains over plain HTTP and rely on
signatures alone.

## 3. Protocol and soundness

Determinism changes the game as follows. Since T is a deterministic function of
C and every input is committed, any published chain ending at M′ ≠ T(C) contains
at least one transition that is not the result of running the committed
computation from its opening boundary. Zero-tolerance replay detects exactly
this: the segment fails, unconditionally. There is no window to hide in — the
entire attack surface of [2] is gone by construction.

**Sampling bound.** An auditor replaying k of N uniform segments catches a chain
with b invalid transitions with probability 1 − C(N−b,k)/C(N,k); for the minimal
forgery b = 1 this is exactly k/N. With m independent auditors publishing
results to the log, detection ≥ 1 − (1 − k/N)^m. Confidence is bought by audit
volume, which is why audit results belong in the transparency log.

**Cost model.** Auditor compute ≈ (k/N) × training cost plus k+1 boundary
downloads. Publisher storage ≈ N × (weights + optimizer state) ≈ 3× weights per
boundary under Adam. Segment length S trades these against each other — and, as
Section 6 shows, against replay-free detection power.

## 4. The audit at scale (positive control)

gpt120m (116,343,808 params, nanoGPT-style, char-level enwik8), 20 segments ×
500 steps, strict deterministic fp32, RunPod secure A40, torch 2.11.0+cu128
(`proofs/chain_a40/`). All 20 segments sealed at a steady 63.6 ± 0.3 s. A k=3
audit (segments {4, 10, 12}, seeded sampling) replayed **bit-exactly**. Measured
economics: auditor 274.5 s vs. prover 1589.0 s → **cost ratio 0.173 vs.
theoretical k/N = 0.15**; the 15% overhead is checkpoint load + tensor hashing
(≈1.44× a prover segment, a constant we reproduced on a second GPU model at
0.460 vs. 0.40). Chain storage: 28.5 GB / 21 boundaries = 1.36 GB per boundary,
matching the 3×-weights prediction. At smoke scale the ratio *exceeds* k/N
(fixed per-segment overhead dominates 3-step segments); the crossover is a
scale effect, and we report both sides of it.

## 5. The hardware equivalence class

One 5×200 gpt120m chain trained on an RTX A4000 was audited with identical
sampled segments from three positions (`proofs/chain_cross/`):

| Auditor | Relation to prover | Verdict |
|---|---|---|
| training pod | same physical GPU | PASS (bit-exact), ratio 0.460 |
| second A4000 pod | same GPU model, different physical GPU | PASS (bit-exact), ratio 0.461 |
| L4 pod, different datacenter | Ampere → Ada | FAIL: closing hash mismatch, both segments |

Verification is **portable across instances of the same GPU model**: an auditor
needs the prover's GPU model and software stack, not the prover's machine —
which makes real-world auditing rentable-by-the-hour. In the cross-architecture
failure the *opening* hashes matched and only the *closing* hashes diverged,
isolating the failure to replay arithmetic (floating-point reduction order)
rather than transfer corruption; the 7.5 GB chain traveled over plain untrusted
HTTP with ed25519 signatures as the only integrity mechanism, by design.

## 6. Replay-free forgery detection and the arms race

Boundaries are full optimizer states, so a forged transition must be consistent
not only with the weights but with the Adam moments it carries. Three O(params)
statistics need no replay: **reach** (max elementwise |ΔW|/(S·lr) — Adam's
per-step update is bounded, so coordinates cannot travel arbitrarily far);
**norm profile** (median/MAD z-score of ‖ΔW‖ across segments — MAD because a
splice inflates a plain std enough to mask itself); **moment consistency**
(cos(ΔW, −m_{k+1}), two-sided envelope). Thresholds are the genuine chain's
widened envelope — zero false positives by construction, and zero observed in
every experiment below. Flagged segments are replayed first, so detection
quality converts into audit efficiency above the k/N floor.

**Lazy forgers** (keep genuine moments), gpt120m, L40S (`proofs/forgery_l40s/`):

| Forgery | S=200 | S=100 | S=10 |
|---|---|---|---|
| alt-seed splice | detected+localized | detected+localized | detected+localized |
| interpolation α=0.10 | detected+localized | detected+localized | detected+localized |
| interpolation α=0.03 | **missed** | detected+localized | detected+localized |
| gaussian edit σ=1e-2 | missed | missed | missed |

**Segment length is a detectability dial**: the stealthy 3% splice hides inside
the reachability envelope at S=200 and is cleanly localized at S≤100. The
mechanism is visible in the genuine moment-cosine range, which rises from
[0.013, 0.050] at S=200 to [0.052, 0.128] at S=10 — segment deltas decorrelate
from Adam's ~(1−β₁)⁻¹-step moment memory as S grows.

**Smart forgers** (forge the moments too, `proofs/smartforger_l40s/`):
*sf-aligned* sets m ∝ −ΔW (defeats a one-sided check; the two-sided envelope
flags it as too consistent, at every S). *sf-calibrated* engineers its moment
cosine into the genuine envelope and copies genuine v: its moment forging
*works* (no moment flag at S≥100), but weight-side detectors catch the splice
regardless, and at S=10 the moment envelope fires as well. *sf-freshv*
fabricates v and trips a hard invariant:

> **Proposition (v-continuity).** For any trajectory produced by Adam with
> second-moment decay β₂, v_{k+1} ≥ β₂^S · v_k holds elementwise across any
> S-step segment. *Proof:* each step sets v ← β₂·v + (1−β₂)·g² ≥ β₂·v; compose
> S times. ∎

With β₂ = 0.999, 82% of v persists across even a 200-step segment; fabricated
moments must satisfy the inequality at every one of ~10⁸ coordinates. The clean
separation result: **moment forging pays only when the weight delta is also
small**, and small-delta forgeries at long segments are precisely the residual
hole that shorter segments shrink and sampled replay prices. Honest limits:
σ=1e-2 gaussian edits evade all replay-free detectors at scale (they are also of
negligible attacker utility, and the final model is checked directly against the
manifest hash); the detectors are audit-prioritization heuristics except where
noted; the sound guarantee remains the replay.

## 7. The execution-variant envelope

Real training uses fused kernels and compilation. Measured (gpt10m, twin runs
per cell, L40S, `proofs/envelope_l40s/`):

| Variant | run-to-run (det on) | run-to-run (det off) | bits vs. eager/manual |
|---|---|---|---|
| eager manual attention | ✓ | ✓ | reference |
| SDPA math | ✓ | ✓ | differ |
| SDPA efficient | ✓ | ✓ | differ |
| SDPA flash | *no fp32 kernel* | — | n/a |
| torch.compile (manual) | ✓ | **✗** | differ |

Every variant is its own bit-universe — even SDPA's *math* backend, nominally
identical arithmetic, differs from manual attention (on GPU and CPU alike). The
execution variant therefore belongs in the commitment, and our chain format
carries it in the signed manifest. The headline positive: **a chain trained
under a committed `compile: true` passes the full segment audit bit-exactly** —
compilation and verification are compatible, provided determinism flags are on
(without them, compiled training is not even run-to-run reproducible).
FlashAttention exposes no fp32 kernel at all: the strict-fp32 recipe cannot use
it, and extending verification into bf16/fp16 flash training is open work.

## 8. Limitations and open problems

(1) **Small-delta/long-segment forgeries** evade replay-free detection and are
caught only with probability k/N per audit. (2) **Pre-commitment data
poisoning** is out of scope. (3) **DDP**: NCCL all-reduce ordering is the known
next cliff; our probe is implemented but unmeasured (2-GPU stock). (4)
**bf16/fp16 and FlashAttention** are outside the verified envelope. (5)
**Reproducibility decay**: the auditor must reconstruct the pinned software
stack years later; environment archival is part of the threat model. (6) The
**lazy-forger boundary**: fabricating moments *bit-consistent with a genuine
run of the committed computation* would require running that computation —
which is no longer a forgery. We conjecture, but have not proven, that no
cheaper strategy defeats zero-tolerance replay plus the moment invariants. (7)
TEE attestation may verify training honesty without replay where the hardware
root of trust is itself trusted; replay verification remains the trustless
alternative.

## 9. Related work

PoL [1]; its break via tolerance-window attacks [2]; safetensors and Sigstore
model-transparency for artifact integrity/identity (orthogonal layers: integrity
≠ identity ≠ process); zkML proof-of-training (sound but intractable at scale);
TEE-based attestation (trusted-hardware assumption). Our contribution is the
middle path: sound-under-sampling process verification with commodity hardware
and zero trust in the prover.

## References

[1] Jia et al., *Proof-of-Learning: Definitions and Practice*, 2021.
arXiv:2103.05633.
[2] Fang et al., *"Proof-of-Learning" Is Currently More Broken Than You Think*,
EuroS&P 2023. arXiv:2208.03567.
