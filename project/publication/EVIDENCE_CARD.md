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
publisher identity tests, and raw source archival work. Raw archives, when present,
are under inventory-addressed `raw/` prefixes with per-archive cards and licenses.
No final source or production precommitment has been signed or anchored yet.

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
