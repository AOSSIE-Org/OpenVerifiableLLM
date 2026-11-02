# OpenVerifiableLLM simple run: a small language model you can check yourself

A 23-million-parameter GPT trained from a seeded random start on one exact pass over English
Wikipedia (2026-09-01 dump), then three passes over a pinned OpenAssistant conversation
selection. Every training checkpoint is published, so anyone can recompute any part of the
training bit for bit and confirm it matches the published record.

Source code: [AOSSIE-Org/OpenVerifiableLLM](https://github.com/AOSSIE-Org/OpenVerifiableLLM)
(the exact code used is also in `code/` here).

## What is in this folder

| File | What it is |
|---|---|
| `conversation-model.pt` | Final model (after the conversation phase) |
| `base-model.pt` | Model after the Wikipedia pass, before the conversation phase |
| `chain.jsonl` | Hash-linked training record: one entry per boundary with the training position, a digest of the full model and optimizer state, and the hash of the previous entry |
| `recipe.json` | Model, batch, learning-rate schedule and seed |
| `ckpt-XXXXX.pt` | Full checkpoint (model, optimizer, position) at every boundary |
| `verifications.jsonl` | Which segments were recomputed, by whom, on which GPU and driver, and the result |
| `release-hashes.txt` | SHA-256 of every model, checkpoint and record file |
| `code/` | Exact code: `simple/ovl.py` (train, verify, chat), the scripts, and the library files it imports |

## How the verification works

Training is split into segments: boundary k to boundary k+1. Each segment can be recomputed on
its own, starting from the published checkpoint k. A segment passes only if the recomputed state
is identical, bit for bit, to the committed entry k+1. Because every segment starts from a
published checkpoint, segments can be checked in any order, by different people, at the same
time. When all segments match, the whole training computation is verified without one long
replay.

Also checkable without a GPU:
- the record's hash links;
- the starting weights, regenerated from the seed and equal to entry 0;
- the published models, equal to the committed states;
- every file, equal to `release-hashes.txt`.

**Coverage at release: all 37 of 37 segments recomputed by the project, all MATCH.** That's 31
Wikipedia segments plus 6 for the conversation phase, recomputed on a second RTX 5090 with a
different NVIDIA driver from training (595.91.07 vs 570.195.03 for segments 0-9). Segment 21 was
also recomputed on a fresh machine using only the public files and `verify_segment.sh`: MATCH. See
`verifications.jsonl` for the GPU, driver and code version of every check.

**How hard is it to forge one segment?** With N segments, M independent checkers who each pick one
random segment catch a single forged segment with probability 1 − ((N−1)/N)^M. One checker who
checks k different segments catches it with probability k/N.

## Check a segment yourself

Requirements: an NVIDIA RTX 5090 and the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(PyTorch 2.8.0, CUDA 12.8). Bit-exact results hold for the same GPU type and software. They were
observed across NVIDIA drivers 570.195.03 and 595.91.07. A different GPU architecture is not
expected to reproduce the bits.

```bash
# download code/ from this folder, then:
bash code/simple/verify_segment.sh                 # one random segment (~20 min)
K=3 bash code/simple/verify_segment.sh             # three random segments
SEGMENTS="7 8" bash code/simple/verify_segment.sh  # chosen segments
```

The script downloads the prepared data (22 GB, hash-checked), the record and only the
checkpoints it needs, then prints MATCH or MISMATCH for each segment.

## Which hardware reproduces the bits (`hardware-map.jsonl`)

The same published segment (32→33) was recomputed with the same software (PyTorch 2.8.0, CUDA 12.8):

| Hardware | Result |
|---|---|
| RTX 5090 (Blackwell, the training GPU type) | MATCH (and all 37 segments on a second 5090) |
| RTX 4090 (Ada) | MISMATCH |
| H100 80GB (Hopper) | MISMATCH |
| x86 CPU (PyTorch 2.14) | MISMATCH |

The seeded starting weights reproduce everywhere tested, including the CPU with a different
PyTorch version (2.14), because they are made on the CPU. Bit-exact
training reproduces on the same GPU architecture: two different NVIDIA drivers (570.195.03 and
595.91.07) gave identical bits. Other architectures run different low-level kernels and do not.
Checkers therefore need an RTX 5090. Anyone can rerun the probe with
`code/simple/hwmap_check.sh` in `release/unlearning-3i-atlas/`.

## Timestamped (`anchor.json`, `anchor.*.tsr`)

`anchor.json` records this release's chain head, the SHA-256 of `chain.jsonl` and of
`release-hashes.txt`, and the dataset commit. It was timestamped by two independent RFC 3161
timestamp authorities (FreeTSA and DigiCert) at 2026-10-20 14:38:43 UTC. A timestamp proves the
release existed unchanged no later than that time, so the record cannot be rewritten afterwards
without detection. It does not prove when training happened. Check it with:

```bash
openssl ts -verify -in anchor.digicert.tsr -data anchor.json -CAfile /etc/ssl/certs/ca-certificates.crt
curl -sO https://freetsa.org/files/cacert.pem; curl -sO https://freetsa.org/files/tsa.crt
openssl ts -verify -in anchor.freetsa.tsr -data anchor.json -CAfile cacert.pem -untrusted tsa.crt
```

## Verifiable unlearning (`release/unlearning-3i-atlas/`)

A fork of this run retrains from checkpoint 30 without one article (3I/ATLAS), with the removal
recorded in its chain and every fork segment checked. It shares checkpoints 0–30 with this
release byte for byte. See that folder's README for the exact leave-one-out effect.

## Data

- Prepared training streams: `prepared/8de1d600…/` in this dataset. Wikipedia: 7,232,582 articles,
  7,118,121,139 training targets, each seen exactly once. Conversation: 3,481 conversations,
  1,278,137 targets.
- Raw inputs: `raw/3a7f5106…/` (the complete Wikipedia dump and the OpenAssistant oasst1 files), and
  the signed source statement `source-commitments/ec263b5c…/`.
- **Rebuild from the raw dump: PASS (2026-10-21).** All six preparation stages (corpus, tokenizer,
  Wikipedia stream, conversation selection, conversation and validation streams) were rerun from
  scratch from the raw dump and the OpenAssistant files. It used the pinned preparation code
  (revision 3f882ad, Python 3.12.14, locked packages), after checking the signed source statement.
  The rebuilt `preparation.json` is byte-identical to the published one: sha256
  `8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68`, the prepared-data ID this
  model was trained on. So the training streams provably come from that Wikipedia dump.
  Reproduce with `code/simple/rebuild_prep.sh` (CPU only; uses about 100 GB of disk, so allow
  150 GB; about a day).

## Training facts

- Model: 6 layers, 384 dimensions, 6 heads, 512-token context, 32,000-token vocabulary, tied
  embeddings. Seed 1234. AdamW, BF16 autocast with FP32 weights, deterministic CUDA kernels.
- Wikipedia: 560,177 updates of 32 × 512 tokens, warmup then cosine learning rate 6e-4 → 6e-5.
  Conversation: 3 passes at 1e-4.
- Code versions: segments 0–18 were trained with `ovl.py` sha256 `202ac6ce…`
  (`code/training-code/ovl-segments-00-18.py`). Segments 19 onward used `c4a942c0…`
  (`code/training-code/ovl-segments-19-end.py`, identical to `code/simple/ovl.py`). The newer version
  only changes CPU-side batch preparation and the CPU thread count. Proof that it produces
  identical results: segments 19 onward were recomputed with the original batch loader and
  matched, and segment 5 (trained with the original code) was recomputed with the newer code and matched.
- The training host restarted once, at boundary 10. Training resumed from checkpoint 10, and
  segment 10 (which spans the restart) matched.

## Limits

- This is a small model. It writes fluent fragments in an encyclopedia style. It often states
  wrong facts and has only basic conversation ability.
- "Verified" here means the published model is exactly what this training on these inputs
  produces. It does not mean the model's answers are true.
- Bit-exact checking needs the same GPU type and software stack as training.
