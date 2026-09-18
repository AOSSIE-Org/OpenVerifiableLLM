# Check retained record, full replay and resume states

`scripts/verify_pilot_cycle.py` checks every retained safe state against an
independently selected pilot record digest and recipe, kernel, stream and code
bindings. It checks the complete record ancestry, complete replay comparison
schedule, selected resume boundary, compatible environment, target and checkpoint
accounting, actual tensor/control hashes and the closed verifier file sets.

This command does **not** rerun arithmetic. Its result explicitly describes
retained-byte and report consistency. Numerical replay remains the separate
`ovl_pipeline.gpu_pilot replay` operation. Neither result is production acceptance,
hardware attestation or independent third-party verification.

Supply canonical JSON with schema `ovl.retained-pilot-cycle-selection.v1` and these
fields: `record_directory`, `record_files`, `binding`, `expected_record`,
`replay_directory`, `replay_files`, `resume_directory`, `resume_files`, and
`resume_from`. Each file inventory contains sorted `path`, `bytes`, `sha256`
entries covering the complete selected directory. The binding uses
`ovl.pilot-record-parent-binding.v1` with `recipe_sha256`, `kernel_sha256`,
`stream_sha256` and `code_root`. Obtain the expected record digest and bindings
from the operator's independently retained selection, not an untrusted report.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/verify_pilot_cycle.py \
  --selection SELECTED_CYCLE.json --selection-sha256 EXPECTED_SELECTION_SHA256 \
  --output FRESH_VERIFICATION.json
```

The checker rejects an existing output and returns a nonzero exit on any missing
or inconsistent evidence. Keep the selected inventories, original files and
verification receipt together. A failed check must not rewrite an earlier result.
