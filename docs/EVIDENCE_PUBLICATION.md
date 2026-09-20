# Immutable evidence publication transport

`scripts/publish_evidence_archive.py` operates only on the existing approved
public AOSSIE evidence dataset. Its closed plans select one complete prepared
dataset, registration signature, progress signature, or safe checkpoint prefix.
It refuses existing prefixes and never deletes files or rewrites history.

Run with `PYTHONPATH=src .venv/bin/python scripts/publish_evidence_archive.py`:

```sh
# Upload only after constructing and inspecting the exact closed inventory.
... upload --plan PLAN.json --staging STAGING --output NEW_UPLOAD_DIRECTORY
# If a request outcome is unknown, reconcile the preserved intent; do not retry.
... reconcile --plan PLAN.json --output EXISTING_UPLOAD_DIRECTORY
# Rehash every anonymously downloaded byte from an immutable public commit.
... download --plan PLAN.json --revision HF_COMMIT --output NEW_DOWNLOAD_DIRECTORY
```

A per-plan OS lease prevents concurrent publishers. Before the remote commit,
the tool saves the selected plan and observed remote parent in a durable intent.
An interrupted or uncertain commit requires read-only reconciliation, followed
by a full fresh download. A matching remote file list is not content verification.
Hard links within the fresh download directory avoid duplicating the downloaded
corpus on disk; every selected file is still transferred and fully hashed.

Transport verification proves only the selected public bytes. The caller must
then authenticate endorsements under independently configured publisher policies,
check prepared manifests or complete safe states, and perform the required raw
reconstruction and numerical replay. Transport receipts cannot admit training or
replace those checks. No production publication has yet used this helper.

Both production and progress signing workflows install the frozen preparation
lock plus `requirements/endorsement.lock`. The additional lock pins run-signature
and safe-state dependencies without modifying the source-committed preparation
environment. The original progress workflow run 35300876849 failed on its missing
PyNaCl import before request selection or signing. The failure and clean local
rebuild evidence are retained in `project/evidence/publication-endorsement-v1/`.
