# The simple run: a fully recomputable Wikipedia model

This is the project's first production result: a 23.1M-parameter GPT trained from a seeded random
start on exactly one pass over the complete eligible English Wikipedia (2026-09-01 dump), then a
pinned conversation phase. Anyone with the same GPU type can recompute any part of the training
bit for bit and confirm the published weights are exactly what this training on these inputs
produces.

- Release: Hugging Face dataset
  [`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence`](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence),
  folders `release/simple-run/` (the model) and `release/unlearning-3i-atlas/` (the unlearning fork).
  The release README there is the user guide; this page is the project record.
- Code: `simple/` in this repository. `simple/ovl.py` trains, verifies and chats in one file.
  The exact code used for the run is also published in `release/simple-run/code/`.

## How it works

- **Data provenance.** The raw dump and the OpenAssistant files are published with checksums and a
  signed source statement. A pinned preparation pipeline turns them into token streams: 7,232,582
  articles and 7,118,121,139 training targets for Wikipedia; 3,481 conversations and 1,278,137
  targets for the conversation phase.
- **Exact coverage.** Every training step checks that its targets continue exactly where the
  previous step stopped. Each phase must end with every target covered once.
- **Determinism.** BF16 autocast with FP32 master weights, deterministic CUDA kernels, no TF32,
  no reduced-precision reductions, a fixed cuBLAS workspace. The learning rate is a pure function
  of the step.
- **Segmented commitments.** A boundary every 18,080 updates, plus the phase boundaries: 38 entries
  and 37 segments. Each entry in `chain.jsonl` records the training position and a SHA-256 over the
  full model and optimizer state, and links to the previous entry's hash. Every boundary checkpoint
  is published.
- **Verification.** Segment k is checked by loading checkpoint k, recomputing to boundary k+1 and
  comparing the full state digest bit for bit. Segments are independent, so they can be checked in
  any order, by different people, at the same time. Checking all of them verifies the whole
  training computation.
- **Timestamps.** `simple/stamp.py` sends every new chain entry to two RFC 3161 timestamp
  authorities (FreeTSA and DigiCert), so the record cannot be rewritten later without detection.
  The release itself was anchored after training. Per-boundary stamping was added after the run.

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
- A mid-run code change to the CPU-side loader and thread pool. Later segments were rechecked with
  the old loader, and an earlier segment with the new one. All matched.
- A driver change (570.195.03 → 595.91.07) gave identical bits.

**Hardware scope** (segment 32→33, PyTorch 2.8.0, CUDA 12.8):

| Hardware | Result |
|---|---|
| RTX 5090 (Blackwell, the training GPU) | MATCH |
| RTX 4090 (Ada) | MISMATCH |
| H100 80GB (Hopper) | MISMATCH |
| x86 CPU (PyTorch 2.14) | MISMATCH |

Bit-exactness holds within one GPU architecture, regardless of driver. Checkers therefore need the
training GPU's architecture. This is the method's main trust assumption.

**Verifiable unlearning.** A fork at boundary 30 removed one article that the run had not yet
reached ("3I/ATLAS", 10,395 targets). The fork shares checkpoints 0–30 byte for byte, records the
removed range in its chain, and its coverage check accepts a gap only at that range. Every fork
segment matched. Loss on the article's text went from 3.3604 to 3.3860 (+0.0256). Mean loss on 50
control articles from the same segment, which both models saw, went from 2.8710 to 2.8712
(+0.0002). One run, no variance estimate.

## Check it yourself

- **No GPU:** check the hash links, the seeded initialization, the published weights and the
  file hashes with `python simple/ovl.py verify --run RUN --weights conversation-model.pt`, and
  `sha256sum -c release-hashes.txt`.
- **One segment on an RTX 5090 (about an hour, about US$1–1.50 rented):** deploy the public RunPod
  template "OpenVerifiableLLM segment check"
  (<https://console.runpod.io/deploy?template=mai1ozh35g>). It runs `simple/checker_pod.sh`
  and prints one result line. To run it by hand: `bash code/simple/verify_segment.sh` from the
  release.
- **Timestamps:** `python simple/stamp.py RUN --check`, or the `openssl ts -verify` commands in the
  release README.

## Timeline

All times UTC.

| When | What |
|---|---|
| July 2026 | First complete deterministic run and segmented replay audit (116M model on A40): same-GPU recomputation matched, other architectures did not |
| September 2026 | 2026-09-01 Wikipedia dump and OpenAssistant files published with a signed source statement; prepared token streams published |
| mid-Sep to 19 Oct | Work on a large production pipeline (rental, transfer and budget guards). No training on real data happened in this period |
| 19 Oct | Switched to one script, `simple/ovl.py`, and segmented verification |
| 19 Oct 15:04 | Training starts on one RTX 5090. A second RTX 5090 mirrors every checkpoint off the training machine and recomputes each segment as it closes |
| 19 Oct ~20:30 | Training host fails; 19 minutes down; resumed from checkpoint 10 |
| 20 Oct ~02:15 | From boundary 19, the CPU thread pool is sized to the container quota: 120 → 63 ms per update |
| 20 Oct 06:46 | Wikipedia pass and conversation phase finished (15.7 hours in total) |
| 20 Oct | All 37 segments MATCH on the second GPU. Release uploaded and hash-checked; anchor timestamped at 14:38:43 |
| 20–21 Oct | Full rebuild of the training data from the raw dump: byte-identical. Hardware map, unlearning fork, fresh-machine checks |
| 21 Oct | Per-boundary timestamps (`simple/stamp.py`) and the one-click checker (`simple/checker_pod.sh` plus a RunPod template); first check: segment 8 MATCH |
| 22–25 Oct | Contributor fixes to `verify` (#213, with #207, #208 and #210 approved); this branch merged into `main` |

## Cost

Prices are RunPod list prices at the time: US$1.19/h for an RTX 5090 and US$0.50/h for the CPU
pod used for the data rebuild. Storage and transfer are not included.

| Item | Time | Cost |
|---|---|---|
| Training (one RTX 5090) | 15.7 h, including a 19-minute outage and about 9 h at half speed before the CPU fix | about US$19 |
| Checking all 37 segments (second RTX 5090) | 10–16 h: one recomputation of every segment, at full or early speed | about US$12–19 |
| Rebuilding the training data from the raw dump (CPU pod) | about 23 h, plus a 3 h 20 min first attempt lost to a host reboot | about US$13 |
| Unlearning fork, hardware map, fresh-machine checks | a few GPU hours on RTX 5090, RTX 4090 and H100 pods | a few US$ |
| Pods kept running until the upload was verified | several hours on two 5090s | the largest avoidable cost |

In total, the release took roughly **US$55–60** of rented compute.

**A clean repeat** at full speed, with no outage and no idle time, would cost about **US$21** for
training plus checking every segment. These numbers come from the measured throughput (63 ms per
update) in `simple/bench_sizes.py`. The same calculation for bigger models:

| Model | Training + full check |
|---|---|
| 23M | about US$21 |
| 42M | about US$34 |
| 70M | about US$52 |
| 110M | about US$73 |

Checking is cheap per person: one segment is about 20 minutes of GPU time. Sampling also helps.
With N segments, M independent checkers who each pick one random segment catch a single forged
segment with probability 1 − ((N−1)/N)^M.

## Things to avoid

- **Don't put a verification method into the goal.** At one point the goal required one long
  continuous replay. Segmented recomputation verifies the same computation, in parallel, at a
  fraction of the risk. The goal should name the property ("every step is checkable"), not the
  procedure.
- **Don't build safety machinery before anything runs.** A month of rental guards, transfer checks
  and budget reservations produced no training on real data. The simple run started the morning
  after the switch and finished in under a day.
- **Don't trust the host's CPU count inside a container.** PyTorch saw 120 cores and started 120
  threads, but the container's quota was 12.75 CPUs. It was throttled every scheduling period, and
  the GPU sat about 50% idle. Read `/sys/fs/cgroup/cpu.max` and size thread pools to it.
- **Don't keep the only copy of a run on container disk.** On RunPod, `/workspace` on container disk
  is wiped when the pod stops. A second machine mirrored every checkpoint as it was written, which
  also made it the natural place to check each segment.
- **Don't expect bits to match across GPU architectures.** Pin the GPU type and the container image
  tag, and say so to checkers up front.
- **Don't let a check pass with nothing checked.** `all([])` is `True` in Python. `verify` must
  fail when it ran no checks (#207).
- **Don't leave paid pods idle.** Copy what you need off, verify it, then terminate. Most of the
  difference between US$21 and the real total was outages, slow early steps and idle time.
- **Don't run day-long jobs without a way to resume.** The first data rebuild was lost to a host
  reboot after 3 h 20 min.

## Lessons learned

- **Run something small end to end first.** July's small complete run worked. The attempt to
  guarantee everything before starting cost a month.
- **Measure the platform.** The single largest speed-up (about 2×) came from a cgroup CPU quota, not
  from the model or the GPU.
- **Verification that tolerates real life is worth more than verification that forbids it.** A host
  failure, a code change and a driver change all happened mid-run. Because every segment is checked
  on its own, each event became one more matched segment rather than a broken proof.
- **Recording the data position makes new things cheap.** Because each chain entry commits to the
  exact training position, forking the run to remove one article took one extra chain entry and a
  partial retrain, and gave an exact leave-one-out measurement for free.
- **Be explicit about what "verified" means.** The model is small and often wrong about facts.
  Verification proves the computation, not the answers.

## Limitations

- A small model: fluent encyclopedic text, frequent factual errors, basic conversation.
- Checking needs the same GPU architecture as training.
- So far all segment checks were run by the project, on separate machines. None has been run by an
  independent third party yet.
- The release timestamp proves the record existed no later than 2026-10-20 14:38:43 UTC, not when
  training happened. Runs started after 21 Oct stamp every boundary as it is written.
- Removal works cheaply only for documents the run had not reached at the fork point. Removing an
  earlier document means retraining from before its first appearance.

## Status of the older pipeline

The `docs/` pages other than this one, and `src/ovl_pipeline/`, describe the earlier production
pipeline. It is frozen: kept for reference and not used by the simple run, apart from the data
preparation and library files it imports. New work should target `simple/`.
