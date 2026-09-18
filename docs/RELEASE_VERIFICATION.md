# Public model releases and verification scope

A final release has two model repositories: the registered attempt's descriptive
AOSSIE base and chat destinations, with an explicit publication identity such as
`release-1`. `release.json` binds both complete payloads,
the loader code revision/root, registration, public source/preparation roots and
an immutable six-file verification-evidence archive. Both repositories carry the
same inventory and Sigstore bundle. Revision pins belong in the later download
receipt, so the signed inventory does not depend on its own future HF commit.

Only an append-only `project/release-commitments/RUN-ATTEMPT-release-N.json` at a single-parent
branch commit that changes **only** that request triggers `release-models.yml`.
Before that commit, separately commit the corresponding canonical
`project/release-policies/RUN-ATTEMPT-release-N.json` with schema
`ovl.release-parent-policies.v1`, `production`, `source`, and ordered `progress`
policies. CI reads this file from the request's parent and requires every selected
signing revision to be a prior ancestor; it does not derive its trust selection
from the downloaded report archive. Consumers still choose policies independently.
The request-only check is defense against accidental mixed changes, not protection
against a branch writer who rewrites the workflow itself. Review the entire pinned
signing commit and its ancestry before choosing its publisher policy. The workflow downloads and verifies
source/registration/progress identities, report-parent relationships and actual
safe base/final checkpoint bytes before endorsing the inventory. It does **not**
rerun raw reconstruction or CUDA training. A valid signature endorses the project
operator's assertions; it is not independent third-party verification.

Before constructing this request, the operator must actually finish the complete
computation verifier described in [COMPLETE_VERIFIER.md](COMPLETE_VERIFIER.md).
Use `production_release.prepare_payloads` with those reports, the exact source statement, public evidence locator and verified exports,
archive all six reports through `scripts/publish_evidence_archive.py`, and use
`production_release.build` to construct the exact signed inventory. The complete
operator context retains the final progress-signing request and its final public
anchor locator/policy. The report builder checks relationships and actual safe
weights; it cannot establish whether an arbitrary saved report was truthfully
produced. Never present that builder or the CI endorsement as computation replay.

Select the expected release statement SHA-256 and exact GitHub signing commit,
workflow identity, repository/owner IDs, branch, hosted-runner identity and Sigstore
trust root **before** accepting the signing bundle. Policies carried in the
public evidence are navigation aids, not an independent source of consumer trust.

## Publish once, recover without overwriting

The payload builder creates each model under `PAYLOADS/base` and `PAYLOADS/chat`,
with the safe five-file model in its `model/` subdirectory, registration, source
notices, preserved GPL-3.0 project license, model card and demonstration receipts.
The publisher rechecks the separate release policy and both actual model payloads
before any remote write. It accepts only new registered-publication AOSSIE names and refuses
an existing repository. It sends no provider key or signing seed to Hugging Face.

Run a read-only destination check **before committing the signing request**:

```bash
PYTHONPATH=src python scripts/publish_models.py check-destinations \
  --statement UNSIGNED_RELEASE.json --output FRESH_DESTINATION_OBSERVATION.json
```

Availability is an observation, not a name reservation. Then obtain and verify the
exact signature before publishing:

```bash
PYTHONPATH=src python scripts/publish_models.py publish --phase base \
  --statement release.json --bundle release.sigstore.json \
  --policy OPERATOR_SELECTED_RELEASE_POLICY.json --payloads PAYLOADS \
  --output FRESH_BASE_PUBLICATION
# Repeat for phase chat with a distinct FRESH_CHAT_PUBLICATION.
```

Creation and commit have separate durable write-ahead records. A successful
creation interrupted before the commit intent can continue with
`resume-unattempted-commit --output EXISTING_PUBLICATION`. An uncertain commit must
use `reconcile --output EXISTING_PUBLICATION`; this only obtains a revision for
fresh download and makes no provider mutation. Uncertain creation has no automatic
ownership inference or retry. Preserve the empty repository and original intent
for explicit investigation. If creation or commit remains uncertain or a name is
occupied, preserve it and select `release-2` (or another unused publication ID).
Rebuild only model cards/payload inventories, repeat destination checks and obtain
a new append-only release endorsement. The source registration, numerical weights,
all training/replay evidence and their roots remain identical. Each publication
identity has its own explicit signing request/policy and fresh destination names.
No operation deletes or replaces an existing release, and no uncertain write is
automatically retried. Record the failed publication and its replacement publicly.

## Fresh download, identity and inference

In an independently trusted checkout of the exact released loader revision,
install the locked CPU verifier dependencies. Supply a separately selected policy
for the release, source and registration and the ordered complete progress policies.
`selection.json` explicitly pins both destinations and full 40-character HF commits:

```json
{
  "base": {"repo": "AOSSIE/openverifiable-RUN-ATTEMPT-release-N-base", "revision": "FULL_HF_BASE_COMMIT"},
  "chat": {"repo": "AOSSIE/openverifiable-RUN-ATTEMPT-release-N-chat", "revision": "FULL_HF_CHAT_COMMIT"}
}
```

All supplied JSON policy/selection files use the pipeline's canonical byte format.
The example above is illustrative; use real lowercase registered identifiers and
immutable revisions. Use `ovl_pipeline.canonical.write_json` to encode selections.

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=src python -m ovl_pipeline.release_download artifacts \
  --selection selection.json --release-policy RELEASE_POLICY.json \
  --production-policy REGISTRATION_POLICY.json --source-policy SOURCE_POLICY.json \
  --progress-policies PROGRESS_POLICIES.json --source-checkout . \
  --output FRESH_DOWNLOAD_CHECK
```

This forces anonymous public downloads at the selected revisions, verifies the
release signature before downloading model weights, rejects unregistered files,
and checks all bytes, actual safe model roots and inference configuration. The
host's optional `.gitattributes` is bounded, downloaded and recorded separately;
it is not part of the signed model payload. Both model copies of the release and
bundle must be identical. All public ancestry signatures and both endpoint safe
checkpoints are freshly downloaded and checked against independent policies.
The fixed public prompts are actually generated twice locally and compared with
the published token IDs. Runtime observations are retained; output compatibility
is checked rather than presumed across different CPUs.

A token mismatch on the same observed runtime records `FAIL`. A mismatch on a
different runtime records `UNSUPPORTED`; neither yields an overall PASS. The
actual and recorded outputs, runtimes and performer labels are retained. CLI exit
status is 0 for PASS, 1 for FAIL and 2 for UNSUPPORTED.

The artifact-mode report says `locally_recomputed_training: false`, reconstruction
and replay `NOT_RUN`. It authenticates operator assertions and artifacts; it
cannot label the operator's saved replay report as computation performed by the
consumer. Generated-answer factual accuracy remains unestablished.

## Complete public-download computation

Use `full` instead of `artifacts`, adding the full raw archive and audited compatible
runtime options below. This downloads every primary checkpoint, then invokes the
actual full computation verifier on **freshly reconstructed** streams. No saved
PASS report, reconstruction cache, checkpoint restore, sampled subset or numerical
tolerance can substitute for the work.

```bash
# Add to the command above, replacing artifacts with full:
# --raw COMPLETE_PUBLIC_RAW_ARCHIVE \
# --lock requirements/gpu.lock --wheels COMPLETE_WHEEL_DIRECTORY \
# --venv COMPATIBLE_GPU_VENV --source src \
# --interpreter-archive PUBLIC_PYTHON_ARCHIVE \
# --interpreter-sha256 EXTERNALLY_SELECTED_SHA256 \
# --interpreter-root UNMODIFIED_PUBLIC_PYTHON \
# --allowed-generated OPERATOR_PINNED_INSTALLER_FILES.json
```

Full mode performs computation before comparing against the publisher's CPU
demonstrations, so an unsupported inference runtime cannot suppress or erase a
completed replay result. Overall PASS still requires every required check. The
raw-input report explicitly records caller-supplied local bytes, full inventory
hashing and anonymous raw download `NOT_RUN` for this command; obtain and verify
the complete public raw archive separately. Matching hashes are not a claim that
this invocation fetched the raw bytes from the network.

The command provisions no resources. Its caller supplies the compatible hardware
and budget guards. A full PASS binds actual fresh reconstruction, regenerated
initialization, continuous all-update state comparisons, both downloaded models,
complete held-out evaluation and repeated greedy inference. A run performed by
this project remains operator verification, never independent third-party credit.
Neither mode makes generated answers factually reliable by proving provenance.
