# Full-data preparation contract and driver

`ovl_pipeline.preparation` orchestrates the complete source-to-token transformation.
Its production entry point requires a canonical `ovl.source-preparation.v2` statement,
a valid public Sigstore bundle, and a separately configured publisher policy. The
identity smoke is not such a statement. No production source commitment has yet been
issued; the implemented driver has only passed synthetic reconstruction tests.

The commitment binds the dated monolithic Wikipedia inventory and official status,
all raw hashes, linked acquisition receipts, OASST repository commit and complete
train/validation files with license/card, exact selection/extraction/tokenizer recipe,
preparation code/workflow inventory, hash-locked environment, and pinned public raw
archive inventory with attribution and a minimum 90-day owner retention target.
Host availability remains best effort; the target is not a hosting guarantee. The implementation is a
closed schema: no manifest-directed commands, Python imports or arbitrary URL fetches.
Acquisition receipts describe operator observations. Checking their identity alone
cannot prove a network history; the driver separately checks complete raw bytes.

## Source signing

The approved Actions workflow has separate identity-smoke and source-commitment jobs.
Only a new canonical JSON file at
`project/source-commitments/<run_id>-<attempt_id>.json` in a single-parent branch
commit triggers source signing. Existing requests cannot be edited, deleted or
reused; unrelated commits do not re-sign an old attempt. The request contains the
complete contract with `source_revision` set to `github-actions-head`. The builder
substitutes the actual clean checkout commit. This avoids a self-referential Git
hash while allowing consumers to reconstruct the exact statement independently.

The builder checks completed official Wikipedia metadata, the pinned OASST public
tree and all declared archive paths. It downloads and hashes every small metadata
and licensing parent; for the three large raw objects it checks public LFS SHA-256
and byte counts. These are explicitly **metadata checks**, not a full public raw
download. Complete anonymous download verification is a separate required operation.
The raw archive must contain `wikipedia/`, `conversation/`, `README.md`, and
`LICENSES.md` with the exact committed inventory. Public tree extras fail closed.

The source job signs using ambient OIDC and performs a CI self-check. Consumers
must reconstruct the statement from the pinned request and known commit, select
their own publisher policy, verify inclusion, then archive and freshly download
the signing evidence. A self-generated CI policy is not external trust. No source
statement authorizes model updates; final production registration remains separate.
The current implementation has only synthetic source-signing admission tests.

Required Wikipedia raw-directory files:

- The complete `enwiki-YYYYMMDD-pages-articles.xml.bz2`.
- `dumpstatus.json`, exactly as retained from the official completed dump.
- The dated `enwiki-YYYYMMDD-index.html`, `-md5sums.txt` and `-sha1sums.txt`.
  Both checksum lists must contain exactly one matching entry for the complete
  monolithic file and agree with the official completion metadata.
- The downloader's original `<filename>.verified.json` and its named immutable
  network receipt. The source commitment pins their hash chain.

The OASST directory contains both pinned Parquet files under `data/`, `README.md`,
`LICENSE`, and `acquisition.json` (the exact retained OASST acquisition receipt).
The receipt's full file inventory, LFS SHA-256 values, Git blob hashes and repository
revision must agree. These metadata files must be copied without changing their bytes.

Install a fresh preparation environment from the trusted source checkout:

```
uv venv --python 3.12.14 .ovllm-cache/preparation-venv
uv pip install --python .ovllm-cache/preparation-venv/bin/python \
  --require-hashes --index-strategy unsafe-best-match \
  -r requirements/preparation.lock
```

Both indices in the lock are explicit official PyPI/PyTorch endpoints, all versions
and artifact hashes are pinned, and the CPU Torch build is explicit. The uv index
strategy flag is needed because the PyTorch index also contains older copies of
unrelated packages. The original failed default-index installation and corrected
clean installation are retained as development evidence. This lock is for CPU
preparation; a GPU training environment will need a separate frozen lock/image.

After the source commitment is issued and published:

```
TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
 .ovllm-cache/preparation-venv/bin/python -m ovl_pipeline.preparation \
 --statement source-commitment.json --bundle source-commitment.sigstore.json \
 --trust-policy independently-selected-policy.json \
 --wikipedia-raw RAW_WIKI --conversation-raw RAW_OASST --output NEW_PREPARED
```

Extraction accounts for every parsed page, including exclusions. It
preserves page/revision attribution and contributor-history URLs in the article
and ledger records. The stripped text is a declared modification; the raw markup
and source licensing notices accompany its public archive. Tokenizer fitting
uses the declared bounded prefix, then encoding covers every eligible article.
Conversation training and official validation streams are separate. Independent
layout/target accounting runs over all three prepared streams. No training occurs.
The resulting `preparation.json` binds every deterministic output manifest/hash.
The time-varying admission observation is stored separately from those output roots.

For full reconstruction, rerun from raw inputs in a fresh output directory and pass
`--compare-preparation ORIGINAL_PREPARED/preparation.json`. Every transformation is
executed again and every derived manifest/hash must match. Production release logic
must bind that expected manifest to its own trusted registration; merely comparing
to a caller-selected file is not final training verification.

Interrupted outputs are preserved and cannot be silently overwritten. To resume
the original preparation, repeat the original command with `--resume` and retain
its sibling `NEW_PREPARED-progress` directory. An exclusive OS lease prevents two
writers. The source commitment (including code/environment/recipe) must be identical.
Raw inputs are fully checked again; completed stages are reused only after their
complete file set, every file hash, saved result and input ancestry validate.
Uncommitted partial stage directories are moved into the sibling progress directory
with their inventory before recomputation. Never delete the sole preserved evidence.

These local recovery receipts are operator caches, not public commitments or proof
of transformation. Full reconstruction must use a fresh output and progress directory
and execute every transformation again. `--resume` with `--compare-preparation` is
refused. A completed-stage corruption fails closed and is preserved for investigation.
A crash inside a stage recomputes that whole stage; XML/tokenization do not resume
inside a file. Adopt any live process before launching another one. Full Wikipedia
runtime, output storage and production reconstruction remain NOT_RUN.

Recovery execution evidence is separate from the deterministic prepared manifests.
Each invocation preserves content-addressed start/completion observations under
`NEW_PREPARED-progress/observations/`, including anchor admission, process identity,
and the exact stages executed or adopted. A killed invocation retains its start
record; it does not acquire a successful completion record. Resumed output equality
alone supplies no clean-reconstruction credit. The verified statement digest is
checked again before the in-memory contract is passed to transformations.

The lease locks the recovery directory inode, and all preparation evidence writes
occur under that lease. Roots are trusted owner-controlled directories: replacing
a root while a process holds it is outside the cooperative-writer model. A dead
process releases its OS lock; process metadata is diagnostic, never authority to
steal a live lock. Stage files, nested directories, the output directory and its
parent are synced before publishing a completed-stage receipt. This assumes a
local filesystem that honors fsync; it is not a hardware power-loss attestation.

Incomplete stages use a durable preservation intent followed by a same-filesystem
atomic rename, with no cross-filesystem copy/delete fallback. Recovery counts every
preserved directory even if interrupted before its completion receipt. At most eight
partial stages may be preserved by automatic retries. Before recomputation it records
actual preserved bytes and free capacity and refuses if replacing even the known
partial size would leave less than 20% filesystem headroom. This is a lower-bound
check, not a prediction of full stage growth; production storage must separately
budget complete preparation, clean reconstruction, and evidence exports. A bound
failure preserves bytes and requires storage reconciliation before retrying.
