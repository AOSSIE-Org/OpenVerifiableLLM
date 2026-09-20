# Operational evidence transport

The existing immutable-prefix publisher now accepts the closed three-file
`operational-evidence` kind: checkpoint.json, retained-export-inventory.json and
retained-exports.tar.gz. The checkpoint byte digest selects the immutable prefix.
This lets complete large operational archives live in the authorized AOSSIE
Hugging Face evidence repository while small observations remain in GitHub.

The same write-ahead intent, repository/parent checks, no-overwrite behavior and
complete anonymous-download verification apply. This is byte transport only:
archive membership, numerical result scope and checkpoint/replay checks remain
separate. Neither upload nor a signed operational receipt verifies training.

Command: `TOKENIZERS_PARALLELISM=false PYTHONPATH=.:src:scripts .venv/bin/python
-m pytest -q tests/test_evidence_publication.py` — 23 passed. Tests use a fake
provider, including actual byte copies, changed downloads, missing/extra files,
uncertain commit recovery and lease exclusion; they supply no public-upload credit.
