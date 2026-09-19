All 22 prepared files (71,294,010,506 bytes) were freshly downloaded anonymously from the exact public revision and checked against whole-file SHA-256 hashes. Failed earlier attempts remain separately preserved. This is an operator download check; clean raw reconstruction and numerical training/replay are separate gates.

The archive contains every response attempt and final receipt; `receipt-inventory.json` selects its complete regular-file contents. Response bodies were hashed in order and discarded; the original local prepared files remain retained.

Reproduce against the selected local prepared inventory using a fresh output directory:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/verify_prepared_public_ranges.py --plan .ovllm-cache/full-prepared-publication-v1/plan.json --plan-sha256 87c1544b5b3786fbf23eaddd812f58b5cffb5605c8050553d0134b579fa1a730 --revision 4f4702ed3845cae2a24a24b8fc7431a048b8fd03 --retained .ovllm-cache/production-preparation-v1 --output /absolute/fresh/verification-directory --maximum-attempts 6
```
