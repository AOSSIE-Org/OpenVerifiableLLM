# Public evidence and private working material

Public reproducibility requires the computation's technical inputs and results.
It does not require publishing personal conversations or AI-agent working context.

Publish explicitly selected source code, dependency specifications, dataset
inventories, transformation definitions, model/training configuration, public keys,
safe checkpoint states, scoped verification reports and release manifests.

Keep private: operator conversations and verbatim instructions, AI prompts and
advisory transcripts, agent/subscription/usage metadata, local handoffs and mutable
progress state, personal account balances, credentials, private signing material,
machine-specific service configuration and unreviewed operational logs.

Do not publish an entire working directory or recursively archive local evidence
merely because it contains some technical results. Use a reviewed allowlist of
files and fields, including archive members. `.gitignore` helps prevent accidental
tracking; it does not remove history or protect files already tracked, and it does
not govern uploads to external artifact stores.

For reports that mix technical results and private details, retain the original
privately and prepare a separately identified public report. Never silently edit a
previously hashed or signed object or represent a sanitized derivative as the
original bytes. Reissue affected inventories and references and check their actual
downloads. Disclose the verified scope and any unavailable historical dependency
without publishing the removed private content.

Publication reviews must cover Git objects, document text, generated reports,
compressed archives, CI artifacts and external datasets. Existing trusted
verification gates still apply to the selected public material.
