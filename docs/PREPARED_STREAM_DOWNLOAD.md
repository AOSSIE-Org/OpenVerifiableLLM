# Complete prepared-data public response verification

`scripts/verify_prepared_public_stream.py` downloads **every byte of every selected
prepared file** at one immutable public Hugging Face revision. It hashes complete
anonymous HTTP responses with bounded memory and keeps per-file receipts. This
avoids allocating another71GB beside the original preparation and its independent
clean reconstruction. It does not use local transport caches, ranges, sampling,
credentials or automatic retries. Original prepared files must still exist and
are fully rehashed before the public download.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/verify_prepared_public_stream.py \
  --plan /absolute/prepared-plan.json --plan-sha256 SELECTED_PLAN_SHA256 \
  --revision IMMUTABLE_40_HEX_REVISION --retained /absolute/original-preparation \
  --output /absolute/fresh-public-response-evidence
```

Response bodies are hashed then discarded; original preparation bytes and public
objects remain retained. The report explicitly says `response_bytes_retained:false`.
This establishes complete public byte identity, not transformation reconstruction,
training replay, signer identity or release verification. Both final model releases
still require the separate retained-download, semantic and inference verification
paths. A failed response keeps a failure record and all completed file receipts;
the original local files are untouched. There is no automatic resumption or PASS
from an incomplete inventory.

Thirteen positive/adversarial CPU tests cover complete streaming, omitted length,
changed/short/oversized bodies, network interruption, partial/encoded responses,
changed inventories, private/moving revisions, wrong plan and missing originals.
The full production-prepared public download remains pending while upload runs.
