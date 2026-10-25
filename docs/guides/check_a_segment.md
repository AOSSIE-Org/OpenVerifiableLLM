# Check a training segment

The released model's training is split into 37 segments. Each segment starts from a published
checkpoint, so anyone can recompute one segment on its own and compare the result, bit for bit,
with the published record. This guide shows three levels of checking, from a laptop to a rented
GPU.

- [What you can check without a GPU](#without-a-gpu)
- [One segment, one click (RunPod)](#one-segment-one-click)
- [One segment, by hand](#one-segment-by-hand)
- [Check the timestamps](#check-the-timestamps)
- [What a check proves](#what-a-check-proves)

The files are in the Hugging Face dataset
[`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence`](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/tree/main/release/simple-run),
folder `release/simple-run/`.

## Without a GPU

Download `release/simple-run/` without the `ckpt-*.pt` files, then from that folder:

```bash
pip install huggingface_hub torch numpy rfc8785==0.1.4 tokenizers==0.23.2 mwparserfromhell==0.7.2 \
  "safetensors>=0.4.5,<1.0" "pynacl>=1.5,<2.0"
sha256sum -c --ignore-missing release-hashes.txt   # every downloaded file matches the hash list
OVL_DEVICE=cpu python code/simple/ovl.py verify --run . --weights conversation-model.pt
# weights conversation-model.pt: MATCH chain entry 37
```

This checks the hash links of `chain.jsonl` and that the published model is exactly the state
committed at the end of training. With the prepared data (`--data`), `verify` also regenerates the
seeded starting weights and compares them with entry 0. That part works on any device, including a
CPU.

## One segment, one click

Recomputing a segment needs an **NVIDIA RTX 5090**, the GPU type the run was trained on. Other
architectures run different low-level kernels and do not reproduce the bits.

1. Deploy the public RunPod template "OpenVerifiableLLM segment check" on an RTX 5090:
   <https://console.runpod.io/deploy?template=mai1ozh35g> (60 GB container disk, no volume).
2. Wait about an hour: a 22 GB download, then about 20 minutes of training.
3. Open the pod's **Logs** or run `tail /workspace/check.log`. The last lines look like:

   ```text
   OVL check PASS: segment 8 MATCH | NVIDIA GeForce RTX 5090, 595.91.07 | torch 2.8.0+cu128 | final state 934d681c99e5 | release 96dc282a | 2026-10-21
   ```

4. Post that line in the project Discord, then **terminate the pod**. It costs about US$1–1.50.

The template runs [`simple/checker_pod.sh`](../../simple/checker_pod.sh) at a fixed commit. It
downloads the release's own code from Hugging Face at a pinned revision and runs
`verify_segment.sh`. Nothing is uploaded, and no account or token is needed. Set `K=3` to check
three random segments, or `SEGMENTS="7 8"` to choose them.

## One segment, by hand

Use an RTX 5090 with the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0,
CUDA 12.8). From the release's `code/` folder, or a clone of this repository:

```bash
bash simple/verify_segment.sh                  # one random segment (~20 min)
K=3 bash simple/verify_segment.sh              # three random segments
SEGMENTS="7 8" bash simple/verify_segment.sh   # chosen segments
```

The script downloads the prepared data (hash-checked), the record and only the checkpoints it
needs. It prints `segment k->k+1: MATCH` or `MISMATCH` for each segment, and appends the result
to `verifications.jsonl`.

## Check the timestamps

`anchor.json` records the release's chain head and file hashes. Two independent RFC 3161 timestamp
authorities stamped it at 2026-10-20 14:38:43 UTC:

```bash
openssl ts -verify -in anchor.digicert.tsr -data anchor.json -CAfile /etc/ssl/certs/ca-certificates.crt
curl -sO https://freetsa.org/files/cacert.pem; curl -sO https://freetsa.org/files/tsa.crt
openssl ts -verify -in anchor.freetsa.tsr -data anchor.json -CAfile cacert.pem -untrusted tsa.crt
```

Runs trained with the current `simple/ovl.py` also stamp every chain entry as it is written.
Check those with `python simple/stamp.py RUN --check`.

## What a check proves

A MATCH means the published checkpoint k, trained under the published recipe on the published
data, produces exactly the published state k+1. If all 37 segments match, the whole training
computation is verified. One forged segment among N is caught by a single random check with
probability 1/N. M independent checkers who each pick one random segment catch it with
probability 1 − ((N−1)/N)^M, which is why every independent check counts.

A check does not show the model's answers are true. It shows the model is exactly what this
training on this data produces.
