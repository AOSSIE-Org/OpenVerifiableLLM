# Public raw archive transport

`scripts/publish_raw_archive.py` publishes only to the authorized AOSSIE evidence
dataset. It adds an inventory-addressed `raw/<SHA256-of-canonical-files-inventory>`
prefix, preserving all previous paths. It pins the current parent commit to prevent
overwriting concurrent work. It does not create repositories or buy storage.

Create a staging directory with only the complete, verified raw Wikipedia source,
its original metadata/receipts, complete OASST splits/card/license/receipt, and the
raw README/LICENSES notices. A separate canonical plan has this closed structure:

```
schema: ovl.raw-archive-plan.v1
repo: AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
prefix: raw/<digest(files)>
files: sorted [{path, bytes, sha256}, ...]
wikipedia_spec: original ovl.download-spec.v1 object
wikipedia_verified: original complete acquisition verified object
```

The `files` inventory includes every archived file, including metadata and notices.
The plan stays outside the uploaded prefix; its digest and exact bytes are retained
in public project evidence. Local hashing checks every file before any upload.

```
PYTHONPATH=src .venv/bin/python scripts/publish_raw_archive.py upload \
 --plan RAW_PLAN.json --staging RAW_STAGE --output NEW_UPLOAD_RECEIPTS
PYTHONPATH=src .venv/bin/python scripts/publish_raw_archive.py download \
 --plan RAW_PLAN.json --revision EXACT_HF_COMMIT --output NEW_DOWNLOAD_DIRECTORY
```

The download uses the public endpoint, `token=False`, `force_download=True`, a fresh
directory/cache, and disables Xet before SDK import. It checks all downloaded files
and independently repeats the entire Wikipedia MD5/SHA-1/SHA-256 and bzip2 integrity
check. Only then does it write `verification.json`. This is operator verification;
it does not establish independent authorship, data transformation reconstruction,
or training replay. A network/signature metadata check is insufficient.

Each command locks its plan and refuses existing output directories. An upload
intent records the original parent, complete inventory, script identity and SDK
version before transmission. If interrupted, inspect the original process/session
and retained intent. Query the intended prefix at the current remote commit: a
completed remote commit may exist even if the response was lost. Download/verify
that exact commit with the original plan to adopt it. Do not upload another copy or
overwrite the prefix. If no commit exists and the original process is terminal,
retain the failed intent and retry with a new receipt directory. Partial downloads
are preserved; restart verification in a fresh directory. No production step may
use a missing or failed verification receipt.

Actual large upload/download feasibility is still pending. The local tests use
synthetic SDK responses and supply no public availability evidence. The live host
preflight is `project/evidence/source-survey/hf-storage-preflight.json`; its storage
API does not report a remaining quota. Public hosting is best effort, with a minimum
90-day owner retention target and no paid add-on authorization.

Upload transport may opt into Hugging Face Xet by setting `HF_HUB_DISABLE_XET=0`
before launching a fresh upload process. This supports chunked upload without
changing the committed file inventory. The intent records whether Xet is disabled.
Download verification still requires Xet disabled and always downloads every file
anonymously into a fresh directory, then checks complete bytes/decompression.
The initial non-Xet upload failed with no committed prefix; original intent, failure
and anonymous remote reconciliation are retained in project/evidence/raw-archive.
