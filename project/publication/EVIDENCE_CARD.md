---
language:
- en
pretty_name: OpenVerifiableLLM Wikipedia provenance evidence (in development)
license: other
license_name: component-specific-licenses
license_link: https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/blob/main/LICENSES.md
---
# OpenVerifiableLLM Wikipedia provenance evidence

**Development in progress. No production-trained or end-to-end verified model is
published here yet.** Synthetic test results do not establish Wikipedia training.

This new AOSSIE repository is reserved for publicly reconstructible inputs,
checkpoints and reports for the OpenVerifiableLLM Wikipedia base model and its
conversational derivative. The governing goal and code are maintained at
[AOSSIE-Org/OpenVerifiableLLM](https://github.com/AOSSIE-Org/OpenVerifiableLLM).

The intended corpus is the complete eligible English article corpus from one
completed dated Wikimedia dump, with a separately pinned OpenAssistant/oasst1
conversation phase. Exact files, revisions, digests, extraction policy and
initialization will be published before production training. Raw reconstruction
and continuous exact replay are required before any verified release claim.

Current scope: publicly downloaded/replayed synthetic development evidence,
publisher identity tests, complete raw source archival and a verified public source/preparation commitment. Raw archives
are under inventory-addressed `raw/` prefixes with per-archive cards and licenses.
G01 source identity/availability is complete: all19files were anonymously downloaded
and rehashed, and the full Wikipedia compressed archive passed decompression and
upstream checksums. The source/preparation statement was signed at
[GitHub commit f6d4371](https://github.com/AOSSIE-Org/OpenVerifiableLLM/commit/f6d43711bf0895c7f90974d1bbfd46ccc4aebda6),
verified against an operator-reconstructed publisher policy, archived at
[HF4b6a102](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/commit/4b6a102ff5046fabdee21a16edaac12f17319b88),
and verified again after a fresh anonymous download. Statement SHA-256:
`ec263b5c3914fd1a8bd25e97fa377b4f8416c38a9ef216d27c203252948b9129`.

Full preparation is running on local CPU (22.36 million source records observed at 09:20 UTC, still incomplete). The [finite coordinator checkpoint](https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/cafd152d86d6c172c06b376978871890a52230e5/project/evidence/finite-coordinator-v1/checkpoint.json) passed 928 local tests with three CUDA skips. A guarded operational diagnostic subsequently found an incompatible host driver and insufficient transfer timing; it was terminated, both guards verified absence, and its billing ceiling remains reserved. This is failed admission evidence, not CUDA training validation. G02–G10 remain pending; no production
training registration, GPU training, full production replay or model release exists.
These checks were performed by the project operator, not an independent third party.
The [public evidence trace](https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/feat/verifiable-wikipedia-pipeline/docs/PUBLIC_EVIDENCE_TRACE.md)
links scoped checks and failed attempts.

## Licensing and retention

Components retain their own source licenses. Wikipedia text and attribution,
OpenAssistant Apache-2.0 data, project source and operator-authored evidence must
be identified separately in LICENSES.md before dataset content is uploaded.
The code license is not a blanket license for Wikipedia text.

The target is at least 90 days of verification-input availability and long-term
retention of final models/manifests under AOSSIE stewardship, subject to confirmed
host policy. This is a retention target, not a guarantee from Hugging Face. No paid
storage add-on, new subscription or permanent GPU service is authorized.

## Verification claims

Artifact integrity, publisher identity, data reconstruction, continuous replay,
and inference reproduction are different checks. Operator-run replay will be
labeled as such. Training provenance does not establish the factual accuracy of
model answers or identify which article caused an answer.
