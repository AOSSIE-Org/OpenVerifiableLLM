# OpenVerifiableLLM: <br/> Language models you can check yourself

<p align="center">
  <img src="docs/images/openverifiablellm_logo.png" width="160" alt="OpenVerifiableLLM logo: a speech bubble with a check mark">
</p>

<p align="center">
  <a href="https://scorecard.dev/viewer/?uri=github.com/AOSSIE-Org/OpenVerifiableLLM"><img src="https://api.scorecard.dev/projects/github.com/AOSSIE-Org/OpenVerifiableLLM/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/15952"><img src="https://www.bestpractices.dev/projects/15952/badge" alt="OpenSSF Best Practices"></a>
  <a href="BestPracticesChecklist.md"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FAOSSIE-Org%2FOpenVerifiableLLM%2Fmain%2Fchecklist-status.json&query=%24.percent&suffix=%25&label=Best%20Practices&logo=openssf" alt="Best Practices"></a>
  <a href="https://discord.gg/hjUhu33uAn"><img src="https://img.shields.io/discord/1022871757289422898?logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=57F287" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License: GPL-3.0"></a>
</p>

When a model is published, you usually have to trust how it was made. OpenVerifiableLLM makes the
training itself the evidence: anyone with the same GPU type can recompute any part of the training
and get exactly the same bits.

**The first release** is a 23.1M-parameter model, trained on one exact pass over all of English
Wikipedia. Its training was saved at 38 points and so split into 37 pieces, called segments. Every
segment was recomputed bit for bit on a second machine. The other terms are defined
[below](#words-used-here).
[Model on Hugging Face](https://huggingface.co/AOSSIE/OpenVerifiableLLM-23M)
· [Data and training record](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence)
· [Releases](https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases) · `ollama run ryoari/openverifiablellm`

<a href="docs/verify.md">
<img src="docs/images/check_snippet.svg" align="right" width="440"
     alt="Terminal session: the published weights match chain entry 37, and a recomputed training segment matches. Follow the link to check one yourself.">
</a>

**What is shown**

-   37 of 37 segments recomputed bit for bit (by the project, on a second RTX 5090)
-   The training data rebuilds byte for byte from the raw Wikipedia dump
-   The record is timestamped by two independent authorities (20 Oct 2026)
-   A fork that provably never trained on one article

**What is not shown**

-   That the model's answers are true. It is small and often wrong
-   That it works on other GPUs. Bits match only on the training GPU type
-   An independent check. No one outside the project has checked a segment yet

<br clear="right"/>

## Words used here

| Word | Meaning |
|---|---|
| **checkpoint** | A saved copy of the full model and optimizer state |
| **boundary** | A point in training where a checkpoint is saved. This run has 38 |
| **segment** | The training between two boundaries. This run has 37. Each one takes about 20 minutes on one GPU |
| **chain entry** | One line of `chain.jsonl`, written at each boundary: the training position, a SHA-256 of the full state, and the hash of the previous entry |
| **chain head** | The hash of the last chain entry. It fixes the whole record |
| **anchor** | `anchor.json`: the chain head and the release's file hashes, timestamped by DigiCert and FreeTSA (RFC 3161) |
| **consistency check** | A check on a laptop: the released files match the timestamped record. It recomputes no training |
| **verification** | Recomputing a segment from its checkpoint and comparing the result bit for bit |

## Getting started

**1. Try the model (2 minutes).** This checks nothing; it is a demo.

```bash
ollama run ryoari/openverifiablellm "Tell me about the Moon."
```

**2. Consistency check, no GPU (10 minutes).** Needs Python 3.10 or newer, `git`, `openssl`,
`sha256sum` and network access. On macOS, `sha256sum` comes with `brew install coreutils` (as
`gsha256sum`).

```bash
pip install huggingface_hub torch numpy
hf download AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence \
  --repo-type dataset --include "release/simple-run/*" --exclude "*/ckpt-*" --local-dir ovl
cd ovl/release/simple-run
sha256sum -c --ignore-missing release-hashes.txt
git clone -q --depth 1 --branch v1.1.0 https://github.com/AOSSIE-Org/OpenVerifiableLLM ovl-code
OVL_DEVICE=cpu python ovl-code/simple/ovl.py verify --run . --weights conversation-model.pt
# anchor anchor.json: chain head MATCH (38 entries)
# anchor timestamp freetsa: OK (Oct 20 14:38:43 2026 GMT)
# anchor timestamp digicert: OK (Oct 20 14:38:43 2026 GMT)
# weights conversation-model.pt: MATCH chain entry 37
```

Use `ovl.py` from the v1.1.0 tag (v1.0.2 or newer works too). The copy in the download's `code/`
folder is older and does not check the anchor.

**3. Verification, on an RTX 5090 (about 1 hour, about US$1.50).**
[Deploy the RunPod template](https://console.runpod.io/deploy?template=mai1ozh35g), wait, and read
the last line of the log. Then terminate the pod. Post the line in Discord: every independent check
counts.

All the details, including macOS and checking by hand: [docs/verify.md](docs/verify.md).

## How it works

```text
 raw Wikipedia dump ──rebuild──▶ prepared tokens ──┐
                                                    ▼
 entry 0 ─▶ entry 1 ─▶ entry 2 ─▶ … ─▶ entry 37     chain.jsonl: each entry hashes the full
   │          │          │                │          model + optimizer state, the data
 ckpt 0     ckpt 1     ckpt 2    …     ckpt 37       position and the previous entry

 check segment k:  load ckpt k ─▶ train to boundary k+1 ─▶ compare with entry k+1, bit for bit
```

Training is deterministic (fixed seed, deterministic CUDA kernels, no TF32). Every step checks that
its data continues exactly where the previous step stopped. So recomputing a segment gives the same
bits, and checking all 37 verifies the whole training. Segments are independent: many people can
check different segments at the same time.

Method, results, cost and lessons: [docs/how-it-works.md](docs/how-it-works.md).

## Repository

| Path | What it is |
|---|---|
| [`simple/`](simple/README.md) | All the code: `ovl.py` (train, verify, fork, chat) and the scripts around it |
| `tests/` | Tests for `simple/`. They run on a CPU |
| [`docs/`](docs/how-it-works.md) | [How it works](docs/how-it-works.md), [verify](docs/verify.md), [roadmap](docs/roadmap.md), the [GSoC 2026 report](docs/gsoc-2026-report.md) |
| [`audit/`](audit/README.md) | Outside reviews and our responses |
| [`security-reviews/`](security-reviews/) | Security reviews and their fixes |
| `requirements/preparation.lock` | The hash-locked CPU environment that CI installs |
| [`brand/`](brand/Brand.md) | Logo, colours and typography |

## Join us

-   **Check a segment.** See step 3 above. It is the most useful contribution right now
-   **Contribute code or docs.** Read [CONTRIBUTING.md](CONTRIBUTING.md)
-   **Talk to us** on the [AOSSIE Discord](https://discord.gg/hjUhu33uAn) or in
    [GitHub issues](https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues)
-   **Report a security problem** privately: see [MAINTAINERS.md](MAINTAINERS.md#reporting-a-vulnerability)

OpenVerifiableLLM is an [AOSSIE](https://aossie.org) project, licensed under [GPL-3.0](LICENSE).
It follows AOSSIE's [code of conduct](https://github.com/AOSSIE-Org/.github/blob/master/CODE_OF_CONDUCT.md).
