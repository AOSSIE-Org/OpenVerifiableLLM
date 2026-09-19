# Reconstructible English Wikipedia and OpenAssistant inputs

This archive preserves exact public inputs for the OpenVerifiableLLM project at
[AOSSIE-Org/OpenVerifiableLLM](https://github.com/AOSSIE-Org/OpenVerifiableLLM).
It supports reproduction, inspection and reuse of the source-to-model process.
These are source inputs and operator acquisition records, not trained models or
proof of verified training. No claim of independent third-party verification is made.

`wikipedia/` contains the unmodified complete monolithic article dump from
[English Wikipedia, 2026-09-01](https://dumps.wikimedia.org/enwiki/20260901/), its
observed completion metadata and complete-file verification receipts. The archive
does not combine monolithic and split copies. The XML/bzip2 format preserves the
published source, including page/revision IDs, titles and original markup.

`conversation/` contains both complete official Parquet splits, the original card
and license of [OpenAssistant/oasst1 at the pinned commit](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b),
and the retained acquisition receipt. Selection for conversational training is a
separately declared transformation; the raw archive includes the original splits.

Use the source commitment's pinned repository revision and inventory to select
bytes. Verify SHA-256 for all objects, Wikimedia's MD5/SHA-1, complete bzip2 integrity,
and linked metadata/receipt parents. Acquisition records are operator observations;
they do not cryptographically prove a historical network transfer. Public LFS
metadata checks do not replace a complete anonymous download.

The owner retention target is at least 90 days for verification inputs and long-term
preservation of final manifests/models. Public storage is subject to the host's
[best-effort policy and limits](https://huggingface.co/docs/hub/storage-limits).
No paid storage add-on has been purchased. Preserve pinned revisions and local
recovery copies; do not rewrite history or remove the only copy of evidence.

See [LICENSES.md](LICENSES.md) before reuse. Corpus provenance is separate from
whether a future model's generated answers are factually correct.
