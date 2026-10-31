# Verify the release

The released model's training is split into 37 segments. Each segment starts from a published
checkpoint, so you can recompute one segment on its own and compare the result, bit for bit, with
the published record. This page gives every way to check, from a laptop to a rented GPU.

The files are in the Hugging Face dataset
[`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence`](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/tree/main/release/simple-run),
folder `release/simple-run/`. The terms (segment, boundary, chain entry, anchor) are defined in the
[README](../README.md#words-used-here).

- [1. Consistency check, without a GPU](#1-consistency-check-without-a-gpu)
- [2. One segment, one click](#2-one-segment-one-click)
- [3. One segment, by hand](#3-one-segment-by-hand)
- [Check the timestamps by hand](#check-the-timestamps-by-hand)
- [What a check proves, and what it relies on](#what-a-check-proves-and-what-it-relies-on)

## 1. Consistency check, without a GPU

Needs Python 3.10 or newer, `git`, `openssl`, `sha256sum` and network access. On macOS,
`sha256sum` comes with `brew install coreutils` (as `gsha256sum`). Download `release/simple-run/`
without the `ckpt-*.pt` files, then from that folder:

```bash
pip install huggingface_hub torch numpy
sha256sum -c --ignore-missing release-hashes.txt   # every downloaded file matches the hash list
git clone -q --depth 1 --branch v1.1.0 https://github.com/AOSSIE-Org/OpenVerifiableLLM ovl-code
OVL_DEVICE=cpu python ovl-code/simple/ovl.py verify --run . --weights conversation-model.pt
# anchor anchor.json: chain head MATCH (38 entries)
# anchor timestamp freetsa: OK (Oct 20 14:38:43 2026 GMT)
# anchor timestamp digicert: OK (Oct 20 14:38:43 2026 GMT)
# weights conversation-model.pt: MATCH chain entry 37
```

`verify` checks four things:

1. Every chain entry links to the hash of the previous one.
2. The chain ends at the head recorded in `anchor.json`.
3. Both timestamps on `anchor.json` verify, for the exact bytes of `anchor.json`. DigiCert is checked
   against the system CA bundle (`/etc/ssl/certs/ca-certificates.crt` on Linux, `/etc/ssl/cert.pem`
   on macOS). FreeTSA's certificates are downloaded from freetsa.org.
4. The weights file is exactly the state at a released entry. `base-model.pt` gives
   `MATCH chain entry 31`; `conversation-model.pt` gives `MATCH chain entry 37`. A `.safetensors`
   copy of either model, from
   [`AOSSIE/OpenVerifiableLLM-23M`](https://huggingface.co/AOSSIE/OpenVerifiableLLM-23M), gives the same
   result (it needs `pip install safetensors`).

It fails if any of these fails, or if `anchor.json` or a timestamp is missing. `--no-anchor` and
`--no-timestamps` skip those checks and print a warning; use them only for a run that has no anchor
yet. Use `ovl.py` from the v1.1.0 tag: the copy in the download's `code/` folder is older and does
not check the anchor.

The exclusion fork (`release/unlearning-3i-atlas/`) has its own anchor, timestamped on 2026-10-31
14:28:00 UTC. That is later than the main run's, because the fork was anchored after the release. The
same command in that folder gives `chain head MATCH (39 entries)` and `MATCH chain entry 38`.

`verify` appends its result to `verifications.jsonl` in the run folder, so it changes that file in
your download. With the prepared data (`--data PREP`), `verify` also checks that its own recipe is the one
the chain was trained with, then regenerates the seeded starting weights and compares them with
entry 0. That works on any device, including a CPU. The
prepared-data check compares each phase's `stream.json` hash with entry 0. It does not hash the
token files; a changed token file is caught when a segment is recomputed.

This is a **consistency check**. It shows that the release matches its timestamped record. It
recomputes no training, and the record comes from the publisher.

## 2. One segment, one click

Recomputing a segment needs an **NVIDIA RTX 5090**, the GPU type the run was trained on. Other
GPU types run different low-level kernels and do not reproduce the bits.

1. Deploy the public RunPod template "OpenVerifiableLLM segment check" on an RTX 5090:
   <https://console.runpod.io/deploy?template=mai1ozh35g> (60 GB container disk, no volume).
2. Wait about an hour: a 22 GB download, then about 20 minutes of training.
3. Open the pod's **Logs**, or run `tail /workspace/check.log`. The last lines look like:

   ```text
   OVL check PASS: segment 8 MATCH | NVIDIA GeForce RTX 5090, 595.91.07 | torch 2.8.0+cu128 | final state 934d681c99e5 | release 96dc282a | 2026-10-21
   ```

4. Post that line in the project Discord, then **terminate the pod**. It costs about US$1–1.50.

The template runs [`simple/checker_pod.sh`](../simple/checker_pod.sh) at a fixed commit. It
downloads the release's own code from Hugging Face at a pinned revision and runs
`verify_segment.sh`. Nothing is uploaded, and no account or token is needed. Set `K=3` to check
three random segments, or `SEGMENTS="7 8"` to choose them.

## 3. One segment, by hand

Use an RTX 5090 with the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0,
CUDA 12.8). From the release's `code/` folder, or a clone of this repository:

```bash
bash simple/verify_segment.sh                  # one random segment (~20 min)
K=3 bash simple/verify_segment.sh              # three random segments
SEGMENTS="7 8" bash simple/verify_segment.sh   # chosen segments
```

The script downloads the prepared data (hash-checked), the record and only the checkpoints it
needs. It prints `segment k->k+1: MATCH` or `MISMATCH` for each segment, and appends the result to
`verifications.jsonl`.

## Check the timestamps by hand

`verify` already does this. To do it yourself with `openssl`:

```bash
# DigiCert. On macOS, use -CAfile /etc/ssl/cert.pem
openssl ts -verify -in anchor.digicert.tsr -data anchor.json -CAfile /etc/ssl/certs/ca-certificates.crt
# FreeTSA
curl -sO https://freetsa.org/files/cacert.pem; curl -sO https://freetsa.org/files/tsa.crt
openssl ts -verify -in anchor.freetsa.tsr -data anchor.json -CAfile cacert.pem -untrusted tsa.crt
```

Each prints `Verification: OK`. Runs trained with the current `simple/ovl.py` also stamp every chain
entry as it is written. Check those with `python simple/stamp.py RUN --check`.

## What a check proves, and what it relies on

A MATCH means: the published checkpoint k, trained under the published recipe on the published
data, produces exactly the published state k+1. If all 37 segments match, the whole training
computation is verified. A check does not show that the model's answers are true.

A check relies on:

- **The same GPU type.** Bits match on the RTX 5090. An RTX 4090, an H100 and a CPU give a
  mismatch. The release can be checked only while RTX 5090s are available.
- **The same toolchain.** Checkers run the same PyTorch, CUDA and container image. A deterministic
  error in that toolchain would match everywhere. That doesn't help a forger, but "verified" means
  "this code on this stack produced these weights".
- **Enough independent checkers.** With N segments, M checkers who each pick one random segment
  catch one forged segment with probability 1 − ((N−1)/N)^M. For N = 37, that is about 110 checks
  for 95% and 169 for 99%. So far there are 0 independent checks.
- **Honest reporting.** A check result is a self-reported line. It is not signed, and nothing proves
  that the checker picked the segment at random. Signed results and segment choice from public
  randomness are on the [roadmap](roadmap.md).
- **Cost.** A full check costs as much compute as the training. Each boundary stores the model and
  optimizer state (284 MB here). For large models, assurance comes from many people each checking a
  random sample.

The timestamps prove that the record existed by 2026-10-20 14:38:43 UTC. They do not prove when or
how the training happened.
