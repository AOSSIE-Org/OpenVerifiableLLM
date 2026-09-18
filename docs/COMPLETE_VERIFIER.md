# Complete computation verification

`ovl_pipeline.production_verify` has one scope: actually reconstruct the complete
raw-to-prepared data path, regenerate initialization and sequentially recompute every
registered update, compare all primary boundaries, and bind both exported models to
that result. It has no option to supply a previous reconstruction/replay PASS report,
resume from prepared cache, restore a prover state, sample updates or relax equality.

Run the coordinator in the separately trusted, hash-locked CPU preparation environment
plus `requirements/endorsement.lock`. The compatible audited GPU environment is a
separate target process on the same host/filesystem. The command itself provisions
nothing; an operating controller must supply resource authorization and cost guards.
A third party can use its own compatible hardware. Our runs remain operator
verification, never independent third-party verification.

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=src TRUSTED_CPU_PYTHON \
  -m ovl_pipeline.production_verify \
  --packet REGISTRATION_PACKET --registration-bundle REGISTRATION_BUNDLE \
  --production-policy EXTERNAL_PRODUCTION_POLICY --source-policy EXTERNAL_SOURCE_POLICY \
  --source-checkout EXACT_SOURCE_CHECKOUT --chain-directory COMPLETE_CHAIN \
  --progress-directory COMPLETE_PROGRESS_ANCHORS --progress-policies EXTERNAL_PROGRESS_POLICIES \
  --raw COMPLETE_PUBLIC_RAW_ARCHIVE --exports BOTH_CANDIDATE_EXPORTS \
  --output FRESH_VERIFIER_OUTPUT \
  --lock requirements/gpu.lock --wheels COMPLETE_LOCKED_WHEELS --venv GPU_VENV --source src \
  --interpreter-archive PUBLIC_PYTHON_ARCHIVE --interpreter-sha256 SELECTED_ARCHIVE_SHA256 \
  --interpreter-root PUBLIC_PYTHON_EXTRACTION \
  --allowed-generated OPERATOR_SELECTED_INSTALLER_FILES
```

The raw root contains the entire pinned archive, including `wikipedia/`,
`conversation/` and retained metadata/licensing files. Every file is rehashed.
The original signed preparation kernel then repeats all decompression, extraction,
tokenizer training, selection and three stream preparations into fresh output.
All six stage roots must equal the registration's prepared manifest; none may be
adopted from cache. The GPU child receives these newly reconstructed streams.

The child performs complete sequential numerical replay under the public interpreter
and wheel audits. The parent checks the resulting model mappings again and evaluates
both models on every eligible official held-out conversation target. Evaluation uses
CPU FP32 cross-entropy, masked targets, FP64 sums and a fixed batch size of one; the
report stores exact float64 byte encodings. Two fixed public greedy demonstrations
are generated twice for each model. Their receipts include input/output IDs, input
hash, model/config roots, decoding settings, software and CPU/interpreter observations.

The result is `ovl.complete-computation-verification.v1`. It explicitly reports
`locally_recomputed: true`, `attested_by: null` and
`independent_third_party: false`. Public-release download verification remains
`NOT_RUN`: that separate check must authenticate the signed release inventory and
check actual downloaded model bytes. Token loss and reproducible demonstrations do
not establish factual accuracy. Checking our signed report is an attestation check;
it does not mean the consumer ran this command.

Incomplete reconstruction, altered inputs, missing replay output, a partial replay,
wrong model mapping, sampled validation or a failed runtime launch cannot produce a
successful complete report. Interrupted outputs are preserved; another complete run
requires a fresh output directory. Budget enough disk for original and reconstructed
prepared data plus both recording and verifier checkpoints before starting.

The implementation's tiny test executes real synthetic raw transformations, CPU
training/replay, safe exports, full held-out evaluation and deterministic inference.
Publisher and GPU-launch adapters are explicit test substitutes. That coverage is
not actual production or independent verification evidence.
