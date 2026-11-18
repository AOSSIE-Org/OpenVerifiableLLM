# How it works

The project's first result is a 23.1M-parameter GPT. It was trained from a seeded random start on
exactly one pass over the complete eligible English Wikipedia (the 2026-09-01 dump), then on a
pinned conversation phase. Anyone with the same GPU type can recompute any part of the training bit
for bit, and so confirm that the published weights are exactly what this training on these inputs
produces.

- [Goal](#goal)
- [Method](#method)
- [Results](#results)
- [Verifiable exclusion](#verifiable-exclusion)
- [What "verified" means](#what-verified-means)
- [Limitations](#limitations)
- [Cost](#cost)
- [Timeline](#timeline)
- [Things to avoid](#things-to-avoid)
- [Questions](#questions)

**Where things are.** The model is in the Hugging Face repository
[`AOSSIE/OpenVerifiableLLM-23M`](https://huggingface.co/AOSSIE/OpenVerifiableLLM-23M) (safetensors and
GGUF). The data and the training record are in the dataset
[`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence`](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence):
`release/simple-run/` (the run, with every checkpoint) and `release/unlearning-3i-atlas/` (the
exclusion fork). The code
is [`simple/`](../simple/README.md): `ovl.py` trains, verifies, forks and chats, and with the model,
data helpers and timestamps it is about 1,100 lines. The exact code used for the run is also
published in `release/simple-run/code/`. The training data was prepared at commit `3f882ad`, which
`simple/rebuild_prep.sh` checks out.

## Goal

A published model should come with everything needed to confirm, without trusting the publisher,
that it is exactly what the stated training on the stated data produces. In priority order:

1. **Every step is checkable.** Each part of training can be recomputed from public files and
   compared bit for bit with a published commitment.
2. **The data is traceable.** The training data can be rebuilt from the raw public source, byte for
   byte.
3. **Coverage is exact.** The record shows which data was trained on, in which order, and that each
   item was seen exactly as claimed, with any exclusion recorded.
4. **Checking is cheap and parallel.** One piece takes about an hour and about a dollar, and many
   people can check different pieces at the same time.
5. **The record can't be quietly rewritten.** It is hash-linked and timestamped by independent
   authorities.
6. **Simple enough to read.** A reviewer can read the training loop in an afternoon.

Not goals: proving that the model's answers are correct; bit-exact results across GPU types; large
models before the method works on small ones; defending against a publisher who fakes a whole,
internally consistent run with unlimited compute. What is in scope: any single forged or changed
step is caught by a cheap independent check.

## Method

- **Data provenance.** The raw dump and the OpenAssistant files are published with checksums and a
  signed source statement. A pinned preparation step turns them into token streams: 7,232,582
  articles and 7,118,121,139 training targets for Wikipedia, and 3,481 conversations and 1,278,137
  targets for the conversation phase.
- **Exact coverage.** Every training step checks that its targets continue exactly where the
  previous step stopped. Each phase must end with every target covered once.
- **Determinism.** BF16 autocast with FP32 master weights, deterministic CUDA kernels, no TF32, no
  reduced-precision reductions, and a fixed cuBLAS workspace. The learning rate is a pure function
  of the step.
- **Segmented commitments.** A boundary every 18,080 updates, plus the phase boundaries: 38 entries
  and 37 segments. Each entry in `chain.jsonl` records the training position and a SHA-256 over the
  full model and optimizer state, and links to the previous entry's hash. Every boundary checkpoint
  is published.
- **Verification.** Segment k is checked by loading checkpoint k, recomputing to boundary k+1 and
  comparing the full state digest bit for bit. Segments are independent, so they can be checked in
  any order, by different people, at the same time. See [verify.md](verify.md).
- **Timestamps.** `simple/stamp.py` sends every new chain entry to two RFC 3161 timestamp
  authorities (FreeTSA and DigiCert). This release was anchored after training; per-boundary
  stamping was added after the run.

## Results

| Measurement | Value |
|---|---|
| Training | 560,177 Wikipedia + 441 conversation updates (batch 32 × 512 tokens) |
| Segments recomputed by the project on a second RTX 5090 | **37 / 37 MATCH** |
| Recomputed from public files only, on fresh machines | segments 8 and 21 MATCH |
| Starting weights regenerated from the seed | MATCH on every device tested, including CPU |
| Release anchor timestamped | FreeTSA and DigiCert, 2026-10-20 14:38:43 UTC |
| Rebuild of the training data from the raw dump | byte-identical `preparation.json` (sha256 `8de1d600…ba68`) |

**Unplanned events that the method survived:**

- A host failure mid-run. Training resumed from checkpoint 10, and segment 10, which spans the
  restart, matched.
- A mid-run change to the CPU-side loader and thread pool. Later segments were rechecked with the
  old loader, and an earlier segment with the new one. All matched.
- A driver change (570.195.03 → 595.91.07) gave identical bits.

**Hardware** (segment 32→33, PyTorch 2.8.0, CUDA 12.8):

| Hardware | Result |
|---|---|
| RTX 5090 (Blackwell, the training GPU) | MATCH |
| RTX 4090 (Ada) | MISMATCH |
| H100 80GB (Hopper) | MISMATCH |
| x86 CPU (PyTorch 2.14) | MISMATCH |

Bit-exactness holds within one GPU architecture, across the two drivers tested (570.195.03 and
595.91.07), with the same PyTorch, CUDA and container image. So checkers need the training
GPU's architecture. This is the method's main trust assumption. The July 2026 study found the same
on a 116M model on A40s.

## Verifiable exclusion

Because each chain entry records the exact training position, a run can be forked and continued
without some data. The release folder for this is `unlearning-3i-atlas`.

- A fork at boundary 30 removed one article that training had not reached yet: "3I/ATLAS", 10,395
  targets.
- The fork shares checkpoints 0–30 byte for byte. Its fork entry keeps the state of checkpoint 30 and
  records the removed range; `verify` checks that entry against checkpoint 30 and the recorded range,
  without training. The coverage check then accepts a gap only at that range. Every retrained
  segment after the fork was recomputed and matched.
- Loss on the article's text went from 3.3604 to 3.3860 (+0.0256). Mean loss on 50 control articles
  from the same segment, which both models saw, went from 2.8710 to 2.8712 (+0.0002). That is about
  100 times more change on the removed article. One run, no variance estimate.

This proves that the fork never trained on the article. It is exact retraining without a range
that training hadn't reached. It is not unlearning in the research sense: it doesn't remove
something a model already learned. Exclusion is by document: other articles that mention 3I/ATLAS
stayed in, and how much of its content they carry hasn't been measured. The 50 controls were matched
for length, not rarity. Both are on the [roadmap](roadmap.md).

## What "verified" means

There are two kinds of check:

- A **consistency check** (on a CPU) shows that the released files match the timestamped record:
  the model is the record's final state, and the record hasn't changed since 2026-10-20. It
  recomputes no training, and the record comes from the publisher.
- A **verification** recomputes one segment from its published checkpoint and compares the result
  bit for bit. Checking every segment verifies the whole computation.

What a verification relies on (the same GPU type, the same toolchain, enough independent checkers,
honest reporting) is in [verify.md](verify.md#what-a-check-proves-and-what-it-relies-on).

## Limitations

- The model is small: fluent, encyclopedia-style text, frequent factual errors, basic conversation.
- Checking needs the same GPU architecture as training.
- So far every segment check was run by the project, on separate machines. No independent third
  party has run one yet. About 110 random independent checks are needed to catch one forged segment
  with 95% probability.
- Check results are self-reported text lines. They aren't signed.
- The release timestamp proves that the record existed by 2026-10-20 14:38:43 UTC, not when or how
  training happened.
- The repository's Git history was rewritten once, in September 2026, before this run. It changed
  the public locations of the earlier pipeline's evidence (see
  [the provenance note](https://github.com/AOSSIE-Org/OpenVerifiableLLM/tree/4aee4a01c2e335e221ae2267a12066b304797c71/project/evidence/history-maintenance-v1/PROVENANCE_STATUS.md)).
  This run's record is anchored by its own timestamps, not by Git history.
- Exclusion is cheap only for documents that training hadn't reached at the fork point. Removing an
  earlier document means retraining from before its first appearance.

## Cost

RunPod list prices at the time: US$1.19/h for an RTX 5090, US$0.50/h for the CPU pod used for the
data rebuild. Storage and transfer are not included.

| Item | Time | Cost |
|---|---|---|
| Training (one RTX 5090) | 15.7 h, including a 19-minute outage and about 9 h at half speed before the CPU fix | about US$19 |
| Checking all 37 segments (second RTX 5090) | 10–16 h | about US$12–19 |
| Rebuilding the training data (CPU pod) | about 23 h, plus a 3 h 20 min first attempt lost to a host reboot | about US$13 |
| Exclusion fork, hardware map, fresh-machine checks | a few GPU hours | a few US$ |
| Pods kept running until the upload was verified | several hours on two RTX 5090s | the largest avoidable cost |

In total, the release took roughly **US$55–60** of rented compute. A clean repeat at full speed (63
ms per update, from `simple/bench_sizes.py`) costs about US$21 for training plus checking every
segment. The same calculation for bigger models: 42M about US$34, 70M about US$52, 110M about US$73.

## Timeline

All times UTC.

| When | What |
|---|---|
| July 2026 | First complete deterministic run and segmented replay audit (116M model on A40): same-GPU recomputation matched, other architectures did not. The audit cost 0.173 of training, against a theoretical 0.15 |
| September 2026 | Wikipedia dump and OpenAssistant files published with a signed source statement; prepared token streams published |
| mid-Sep to 19 Oct | Work on a large production pipeline (rental, transfer and budget guards). No training on real data in this period |
| 19 Oct | Switched to one script, `simple/ovl.py`, and segmented verification. Training starts at 15:04 on one RTX 5090; a second RTX 5090 mirrors every checkpoint and recomputes each segment as it closes |
| 19 Oct ~20:30 | Training host fails; 19 minutes down; resumed from checkpoint 10 |
| 20 Oct ~02:15 | From boundary 19, the CPU thread pool is sized to the container quota: 120 → 63 ms per update |
| 20 Oct 06:46 | Training finished (15.7 hours in total) |
| 20 Oct | All 37 segments MATCH on the second GPU. Release uploaded; anchor timestamped at 14:38:43 |
| 20–21 Oct | Data rebuild: byte-identical. Hardware map, exclusion fork, fresh-machine checks |
| 21 Oct | Per-boundary timestamps and the one-click checker; first check: segment 8 MATCH |
| 26 Oct | v1.0.0: GGUF export and Ollama |
| 28 Oct | v1.0.1: `verify` fails without the anchor. Peer evaluation of the release |
| 31 Oct | v1.0.2: `verify` checks the timestamps itself, accepts the released base model, and ties each entry's model digest to its weights. The old pipeline is removed from `main`. The exclusion fork is anchored and timestamped |
| 1 Nov | v1.1.0: all the code in `simple/`, the docs in five pages, safetensors. The model gets its own Hugging Face repository |

## Things to avoid

- **Don't put a verification method into the goal.** At one point the goal required one long
  continuous replay. Segmented recomputation verifies the same computation, in parallel. Name the
  property ("every step is checkable"), not the procedure.
- **Don't build safety machinery before anything runs.** A month of rental guards, transfer checks
  and budget reservations produced no training on real data. The simple run started the morning
  after the switch and finished in under a day.
- **Don't trust the host's CPU count inside a container.** PyTorch saw 120 cores and started 120
  threads, but the container's quota was 12.75 CPUs. It was throttled in every period, and the GPU
  sat about 50% idle. Read `/sys/fs/cgroup/cpu.max` and size thread pools to it.
- **Don't keep the only copy of a run on container disk.** On RunPod, container disk is wiped when
  the pod stops. A second machine mirrored every checkpoint, which also made it the natural place to
  check each segment.
- **Don't expect bits to match across GPU architectures.** Pin the GPU type and the container image,
  and tell checkers up front.
- **Don't let a check pass with nothing checked.** `all([])` is `True` in Python. `verify` fails
  when it ran no checks.
- **Don't leave paid pods idle.** Copy what you need, verify it, then terminate.
- **Don't run day-long jobs without a way to resume.** The first data rebuild was lost to a host
  reboot after 3 h 20 min.
- **Run something small end to end first, and measure the platform.** The biggest speed-up (about
  2×) came from the CPU quota, not from the model or the GPU.

## Questions

**Isn't checking as expensive as training?** Checking everything costs about as much as training
once. But nobody has to check everything: each segment takes about 20 minutes on one GPU, and many
people checking one random segment each catch a forged segment with high probability.

**Couldn't the publisher fake the whole run?** A segment that wasn't really computed from its
published checkpoint and the published data fails the first check that picks it. So a convincing
fake would have to be a real training run on the published data, which is the thing being claimed.
A check can't rule out a segment that nobody has checked yet. That is why independent checks matter.

**Can a model "forget" a document?** Not in the sense of removing something it already learned.
What the method shows is that a model never trained on a document. See
[Verifiable exclusion](#verifiable-exclusion).

**Is the model any good?** It's small. It writes fluent, encyclopedia-style text, often gets facts
wrong, and holds a basic conversation. The point is the method, which works the same way for bigger
models.

**What happened to the older pipeline?** The project first built a large production pipeline with
many safeguards around renting machines and moving checkpoints. It never trained on real data. The
simple run replaced it. The old code, its docs and the July 2026 study were removed from `main` and
are kept in the history at
[`4aee4a0`](https://github.com/AOSSIE-Org/OpenVerifiableLLM/tree/4aee4a01c2e335e221ae2267a12066b304797c71).
