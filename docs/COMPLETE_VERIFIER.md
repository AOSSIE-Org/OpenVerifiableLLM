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

## Joining separately executed operator work

An operator may complete the clean data reconstruction before renting a GPU, then
perform the complete continuous replay later. This preserves the same numerical
and corpus requirements while avoiding a second reconstruction solely to place
both operations inside one command. Use the fresh-work command above to execute
your own complete verification.

`scripts/assemble_computation_evidence.py` produces an explicitly different
`ovl.complete-computation-verification.v2` report from those retained executions:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=src:scripts TRUSTED_CPU_PYTHON \
  scripts/assemble_computation_evidence.py \
  --selection ASSEMBLY_SELECTION.json --selection-sha256 EXTERNALLY_SELECTED_SHA256
```

The caller selects the registration and publisher policies, immutable public
execution evidence, complete original reconstruction observation, audited replay
launch and successful worker termination, complete retained output inventories,
raw/prepared data, recorded and verifier states, and both exports. The command
authenticates the registration and all public primary boundaries again, downloads
and checks the selected public execution documents, rehashes every raw/prepared
input and retained output, compares every required primary and recovery safe
state, verifies both export mappings, and executes complete held-out evaluation
and fixed inference checks. It rejects partial replay, adopted reconstruction
stages, altered bytes, missing audits, failed exits and disconnected parents.

The earlier reconstruction must itself have freshly executed all six stages;
neither sampled transformations nor cached preparation qualifies. Its original
execution report and timing remain identified separately. `total_ms` measures
the assembly command; `reconstruction_and_input_validation_ms` retains the
earlier complete reconstruction duration. The report uses `locally_recomputed:
false`, names the separate operator executions, and explicitly records zero new
numerical updates and no new raw transformations during assembly. It does not
claim that matching saved states proves their training trajectory. The original
audited continuous replay execution supplies the operator's execution evidence;
the retained process observations are not hardware attestation.

Both report versions retain all existing release coverage, ancestry, model,
evaluation and scope checks. A release using v2 also explains the separate
executions in its model cards. Neither assembly, a signed report, nor a public
download constitutes independent third-party recomputation.

## Historical preparation with frozen production code

When the signed preparation inventory predates the registered production tree,
use the separately pinned `scripts/verify_complete.py` driver. The older
`ovl_pipeline.production_verify` command above imports preparation from the current
production tree and rejects that historical inventory. It is unsuitable for this
combination; a signature or the historical contract smoke check does not repair it.

Select an immutable reviewed driver revision independently of the model packet.
Keep three identities separate: the original signed preparation inventory, the
registered production/replay/loader revision, and the new driver revision. Obtain
the driver scripts from the approved source repository at that exact revision.
Run with the frozen production checkout on `PYTHONPATH`; do not copy newer modules
into it. The driver checkout may be separate from the frozen checkout:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=FROZEN_CHECKOUT/src TRUSTED_CPU_PYTHON \
  DRIVER_CHECKOUT/scripts/verify_complete.py \
  --source-checkout FROZEN_CHECKOUT \
  --source FROZEN_CHECKOUT/src \
  --lock FROZEN_CHECKOUT/requirements/gpu.lock \
  OTHER_REQUIRED_OPTIONS_FROM_THE_COMPLETE_COMMAND_ABOVE
```

The remaining options are identical to the full command above. All directories
must be supplied explicitly; the uppercase names are placeholders. This driver
provisions nothing. It materializes only the reviewed closed historical inventory
from the retained clean Git revision, authenticates its bytes before import, and
runs the original `prepare_committed` in a fresh isolated CPU child. The original
publisher policy remains mandatory; the retained revision is a byte supplier, not
a replacement signing revision. Python and installed CPU dependencies remain
trusted. Isolated mode is not a sandbox or runtime attestation.

All six transformations execute into fresh output; no stage adoption is allowed.
The unchanged audited GPU launcher then executes complete sequential replay from
regenerated initialization. The driver checks launch/session parents, complete
coverage, both exports, every held-out target and repeated inference. Its exact
three script hashes are retained separately in `driver.json`; the existing v1
computation report retains its original schema and truthful execution scope.

For both freshly downloaded releases, use
`scripts/verify_release_complete.py full` with the same arguments as the older
`ovl_pipeline.release_download full` command. The external driver performs actual
anonymous downloads, independent release/parent authentication, the repaired fresh
computation, model-root comparisons and downloaded inference. `FAIL` and
`UNSUPPORTED` inference results remain distinct and are never promoted to PASS.
The frozen `release_download artifacts` command remains valid for its narrower
identity-and-operator-evidence scope.

Before signing release inventories, model cards must link to the exact immutable
revision of this corrected guide and the independently selected driver, alongside
the frozen loader revision. Changing a card after signing invalidates its inventory.
Synthetic bridge tests are software evidence only. Reusing a previously completed
full operator reconstruction with a later full replay remains the separate v2
assembly route described above; it must not claim a fresh full-driver execution.
