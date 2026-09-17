# Full-data preparation contract and driver

`ovl_pipeline.preparation` orchestrates the complete source-to-token transformation.
Its production entry point requires a canonical `ovl.source-preparation.v1` statement,
a valid public Sigstore bundle, and a separately configured publisher policy. The
identity smoke is not such a statement. No production source commitment has yet been
issued; the implemented driver has only passed synthetic reconstruction tests.

The commitment binds the dated monolithic Wikipedia inventory and official status,
all raw hashes, linked acquisition receipts, OASST repository commit and complete
train/validation files with license/card, exact selection/extraction/tokenizer recipe,
preparation code inventory, and hash-locked environment. The implementation is a
closed schema: no manifest-directed commands, Python imports or arbitrary URL fetches.
Acquisition receipts describe operator observations. Checking their identity alone
cannot prove a network history; the driver separately checks complete raw bytes.

Required Wikipedia raw-directory files:

- The complete `enwiki-YYYYMMDD-pages-articles.xml.bz2`.
- `dumpstatus.json`, exactly as retained from the official completed dump.
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

Extraction accounts for every parsed page, including exclusions. Tokenizer fitting
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

Interrupted outputs are preserved and cannot be silently overwritten. This first
driver requires a fresh output directory to retry preparation; stage-level resume
is not yet implemented. Adopt any live process before launching another one. The
raw downloader and training fixture have separate tested resume mechanisms. Full
Wikipedia runtime, output storage and production reconstruction are still NOT_RUN.
