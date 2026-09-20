# Complete public prepared-data verification through bounded responses

The initial whole-response verifier rejected the large article file when its HTTP
response ended at 11,807,882,767 bytes instead of 33,012,803,655. Its twelve prior
file receipts and failure remain under `project/evidence/prepared-public-download-v1-failure`.
No complete-inventory success was issued. Original prepared files remain intact.

`scripts/verify_prepared_public_ranges.py` reads every selected byte in contiguous
ranges, with at most four concurrent responses of at most 64 MiB each. It checks
the requested start/end, representation size, length and encoding. A partial
response must have status206 and its exact `Content-Range`; status200 is allowed
only when the requested interval is the entire file. These checks follow the
[HTTP range semantics in RFC9110](https://www.rfc-editor.org/rfc/rfc9110.html#section-14.4).

Completed responses enter one SHA-256 in file order, irrespective of transfer
completion order. Each full-file length and SHA-256 must equal the original pinned
inventory. Range hashes alone never pass a file. The verifier rehashes all retained
original files, requires the exact public revision and complete remote inventory,
and uses anonymous requests without a local transport cache.

Short responses and transient transport failures receive at most one new attempt
for the same interval. Both attempts have separate immutable receipts. Failed
bytes do not enter the full-file hash. Incorrect headers, oversized responses or
whole-file mismatches fail closed. A response has a fixed600-second lifetime,
checked around reads that return available bytes, plus at most one pending
60-second socket operation. No retry renews a paid rental or watchdog deadline.

The fresh output directory is once-only. A process interruption does not permit
adopting an opaque partial hash state or claiming complete coverage. Response
bodies are discarded after hashing; originals and all response receipts remain.
This is complete public byte-identity checking, not raw reconstruction, numerical
replay or independent third-party verification.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/verify_prepared_public_ranges.py \
  --plan SELECTED_PREPARED_PLAN.json --plan-sha256 EXPECTED_PLAN_SHA256 \
  --revision IMMUTABLE_HF_COMMIT --retained ORIGINAL_PREPARED_DIRECTORY \
  --output FRESH_RECEIPT_DIRECTORY
```

The result schema is `ovl.prepared-range-download-verification.v1`. Consumers must
explicitly bind it to the selected full plan, revision, byte count and inventory;
it does not masquerade as the earlier single-response result.
