# OpenVerifiableLLM: a language model you can recompute bit for bit

GSoC 2026 final report. Rajat Roy ([@ryoari](https://github.com/ryoari)), AOSSIE, mentored by Archit Ojha
([@Archit381](https://github.com/Archit381)).

- Code: https://github.com/AOSSIE-Org/OpenVerifiableLLM (everything that matters is in `simple/`)
- Model (safetensors and GGUF): https://huggingface.co/AOSSIE/OpenVerifiableLLM-23M
- Data and training record: https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
- Releases: https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases
- Try it: `ollama run ryoari/openverifiablellm`

## The idea

When someone publishes model weights, you have to take their word for how the weights were made: which
data, which code, which settings. "Open weights" lets you run the model. It doesn't let you check it.

But training is just a computation. If it's deterministic, then anyone with the same code, the same data and
the same kind of GPU should get exactly the same bits. So the claim "these weights came from this data" stops
being a promise and becomes something you can recompute. That is the whole project: make one real training
run deterministic enough that a stranger can recompute any part of it and compare hashes.

## What I built

A 23M-parameter GPT, trained on exactly one pass over all of English Wikipedia (the 2026-09-01 dump,
7,232,582 articles, about 7.1 billion training tokens), then a short conversation phase. It's small, it's
often wrong about facts, and that's fine. The point is the method, not the model.

The training loop is one script, `simple/ovl.py`. It trains, verifies, forks and chats. With the model and
a few helpers it's about 1,100 lines.

How it works:

1. **The data is pinned.** The raw dump is published with checksums. A preparation step turns it into token
   streams, and anyone can rerun that step and get the same bytes. We did, on a fresh machine: byte-identical.
2. **Every step is checked for coverage.** Each training step checks that its tokens continue exactly where
   the previous step stopped. No article is skipped, none is seen twice.
3. **The run is deterministic.** Seeded initialization, deterministic CUDA kernels, no TF32, a fixed cuBLAS
   workspace, and a learning rate that is a pure function of the step.
4. **The run is cut into 37 segments.** At 38 points along the way it saves a checkpoint and writes a line
   to `chain.jsonl`: where training is, a SHA-256 of the full model and optimizer state, and the hash of the
   previous line. Like a tiny blockchain, but boring.
5. **Anyone can check any segment.** Load checkpoint k, train to k+1, compare the hash. One segment is about
   20 minutes on an RTX 5090, about a dollar rented. Segments are independent, so many people can check
   different ones at the same time, in any order. There's a one-click RunPod template that prints one line to
   share.
6. **The record is timestamped.** Two independent RFC 3161 authorities (DigiCert and FreeTSA) stamped the
   final record on 2026-10-20. Since v1.0.2, `ovl.py verify` checks both stamps itself.

## What happened

- All 37 segments were recomputed on a second RTX 5090: 37 of 37 MATCH, bit for bit.
- A host failure in the middle of training: it resumed from checkpoint 10, and the segment across the restart
  still matched. A mid-run code change and a driver change also left the bits identical.
- Same software on other hardware: RTX 4090, H100 and CPU all MISMATCH. Bit-exactness holds within one GPU
  architecture. That's the main trust assumption, and it's stated up front.
- **Verifiable exclusion.** Because every checkpoint is saved, you can go back and take a different path.
  I forked at checkpoint 30 and removed one article (3I/ATLAS) that training hadn't reached yet. The fork
  shares checkpoints 0–30 byte for byte, records the removed range, and the coverage check refuses any other
  gap. So "this model never trained on that article" is something you can check, not just claim. Loss on the
  removed article went up 3.360 → 3.386; on 50 control articles it barely moved (2.8710 → 2.8712). One run,
  so don't read much into the exact size.
- Released as v1.0.0 (26 Oct), with a GGUF export that runs in Ollama.
- Cleaned up in v1.1.0 (31 Oct): the old pipeline is gone from `main`, all the code is in `simple/`, the docs
  are five short pages, and the model has its own Hugging Face repo with safetensors files. The fork's
  record is timestamped too (31 Oct).

## What "verified" means here, and what it doesn't

It means: this exact code, run on this exact data, produces exactly these weights, and the record hasn't
changed since 20 October. It doesn't mean the model's answers are true, that Wikipedia is right, or that it
works on any GPU. And today every check was done by the project, on a second machine. To catch a single
faked segment 95% of the time, you'd want about 110 random checks by people who don't know me. That number
is 0 so far. Getting it above 0 is the most useful next step.

## Outside review

A peer evaluation ([audit/](https://github.com/AOSSIE-Org/OpenVerifiableLLM/tree/main/audit)) found two real
problems: the released base model failed `verify` (the phase-change entry repeats its digest), and a forged
record could pass the CPU check if you skipped the separate timestamp step. Both are fixed in v1.0.2, and the
forged copy now fails. A contributor separately found that segment replay didn't tie the chain's model digest
to the replayed weights (#263).

## What I'd tell myself in March

- **Run something small end to end first.** A small complete run in July worked. Then I spent about a month
  building a large production pipeline with guards around renting machines and moving checkpoints, trying to
  guarantee everything before starting. It never trained on real data. The simple run replaced it with one
  script and finished training in under a day.
- **Don't write the method into the goal.** At one point the goal said "full continuous replay". Segmented
  recomputation verifies the same computation, and it's parallel.
- **Measure the platform.** The biggest speed-up wasn't the model or the GPU. A 120-thread CPU pool on a
  12.75-CPU cgroup quota kept the GPU about half idle. Sizing the pool to the quota roughly halved the time per
  update.
- **Say exactly what was checked, by whom.** Most of the work in the last week was wording: "verified" vs
  "correct", "consistency check" vs "actual verification", "the project" vs "independent".

## What's left

- Independent checks. The one-click checker exists; it needs people.
- Signed check results, and segment choice from public randomness, so results can't be faked in bulk.
- Cheaper checks: per-segment data slices, so a checker doesn't download 22 GB.
- Near-duplicates: other articles that mention 3I/ATLAS stayed in. Exclusion by topic, with the filter itself
  recorded in the chain, is the harder and more useful version.

## Work

117 merged pull requests, March–October 2026. The ones that shaped the result:

- [#34](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/34) development environment (uv, ruff)
- [#73](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/73) checkpoint reproducibility and entropy sources in PyTorch
- [#92](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/92) reproducibility matrix across seeds and hardware, verify before load
- [#217](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/217) the simple run: one-script verifiable training loop
- [#236](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/236) one-click segment check pinned to the release
- [#237](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/237) GGUF export and Ollama
- [#239](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/239) security review and its fixes
- [#252](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/252), [#267](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/267) `verify` requires the anchor and checks its timestamps
- [#253](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/253) what "verified" means, and the limits
- [#271](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/271), [#272](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/272), [#273](https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/273) cleanup: one folder, five doc pages, safetensors

Full list: https://github.com/AOSSIE-Org/OpenVerifiableLLM/pulls?q=is%3Apr+author%3Aryoari+is%3Amerged

Thanks to Archit for the mentoring and for pushing me to keep it simple, to the AOSSIE admins, to the
contributors who reviewed and fixed things, and to dhruvi-16-me for the end-to-end evaluation.
