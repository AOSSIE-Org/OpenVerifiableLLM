# Near-duplicate and topic exclusion

The exclusion fork (`release/unlearning-3i-atlas`) removed the article "3I/ATLAS" as one document.
Because other articles that mention the comet remained in the training stream, the model could still
learn facts about it from those secondary mentions. This page describes how secondary mentions are
detected, how residual loss is measured against rarity-matched controls, and how topic-level
filter specifications are generated and verified.

- [Why document exclusion leaves residual facts](#why-document-exclusion-leaves-residual-facts)
- [Scanning secondary mentions](#scanning-secondary-mentions)
- [Rarity-matched controls](#rarity-matched-controls)
- [Passage loss and token attribution](#passage-loss-and-token-attribution)
- [Bootstrap significance test](#bootstrap-significance-test)
- [Topic filter specification](#topic-filter-specification)
- [Usage](#usage)

## Why document exclusion leaves residual facts

When training forks at boundary 30 to remove "3I/ATLAS", 10,395 training targets from that document
are skipped. However, Wikipedia contains mentions of the comet in other articles (such as
astronomical discovery catalogues, interstellar object lists, and survey reports).

If these secondary passages remain in the training stream, the fork only tests document removal, not
concept removal. To test concept removal, secondary mentions must be quantified and evaluated.

## Scanning secondary mentions

[`simple/near_duplicates.py`](../simple/near_duplicates.py) scans `wikipedia/documents.jsonl` and
`tokens.u16` for entity mentions:

1. **Token patterns.** Entity names and aliases (including leading-space variants required by BPE
   tokenization) are encoded and shifted by `data.OFFSET` (5 special tokens).
2. **Subsequence matching.** `find_token_subsequences` runs a vectorized sub-array match over the
   `uint16` token stream.
3. **Training chunk alignment.** Mentions are aligned to exact training chunk boundaries ($k \times
   \text{ctx}$, with context length 512). This matches the exact token windows seen during training.
4. **Document filtering.** The primary excluded document is filtered out by page ID so only
   secondary mentions are evaluated.

## Rarity-matched controls

In `simple/loo_eval.py`, controls were picked randomly from the same segment based only on document
length. Because rare astronomical objects have different baseline loss characteristics than common
prose, `near_duplicates.py` compares target mentions against passages mentioning equally rare
astronomical controls:

- `1I/'Oumuamua`
- `2I/Borisov`

This isolates entity-specific unlearning from general training drift across checkpoints.

## Passage loss and token attribution

For each detected window, the script runs a forward pass under both models:

- Original base model: $\mathcal{L}_{\text{orig}}$
- Fork base model: $\mathcal{L}_{\text{fork}}$

It reports:
- The full window loss delta: $\Delta_{\text{win}} = \mathcal{L}_{\text{fork}} - \mathcal{L}_{\text{orig}}$
- The mention span loss delta: $\Delta_{\text{mention}}$
- Token-by-token loss attributions around the mention, identifying which specific tokens experienced
  increased next-token perplexity after document removal.

## Bootstrap significance test

Rather than reporting a point estimate without variance, `near_duplicates.py` computes:

1. **Paired bootstrap confidence interval.** 1,000 resamples of target deltas versus control
   deltas compute an empirical 95% confidence interval for net unlearning.
2. **Permutation test.** A two-sample permutation test under the null hypothesis $H_0$: target delta
   $\le$ control delta, yielding an empirical $p$-value.

A net increase in loss on secondary mentions with $p < 0.05$ indicates that removing the primary
article also weakened knowledge in surrounding corpus mentions.

## Topic filter specification

To enable excluding every mention across the corpus rather than just a single document, `near_duplicates.py`
generates a cryptographically verifiable topic filter specification (`topic_filter.json`):

```json
{
  "filter_name": "topic_near_duplicate_filter",
  "version": "1.0.0",
  "filter_code_sha256": "4a7b...",
  "tokenizer_sha256": "9b1c...",
  "target_entity": "3I/ATLAS",
  "aliases": ["C/2025 N1"],
  "matching_patterns": ["3I/ATLAS", " 3I/ATLAS", "C/2025 N1", " c/2025 n1"],
  "excluded_ranges_count": 14,
  "total_excluded_targets": 7168,
  "ranges": [[10523400, 512], [14982100, 512]],
  "sample_miss_rate": null,
  "sample_file_sha256": null
}
```

The filter records:
- The SHA-256 digest of the entire `simple/near_duplicates.py` source code (normalized to LF line endings).
- The SHA-256 digest of the tokenizer definition (`tokenizer.json`).
- Explicit entity aliases passed via `--alias`.
- Merged, strictly sorted, non-overlapping target token ranges to exclude.
- Optional hand-checked sample validation metrics and sample file SHA-256 (`--sample-check`).
- The deterministic SHA-256 digest of the specification itself.

This specification serves as a standalone verifiable input artifact for subsequent topic forks and dataset
filtering tools; it is intentionally not directly hooked into the core training loader (`simple/ovl.py`),
keeping the training loop minimal and cleanly separating measurement artifacts from training execution.

## Usage

### Evaluate near-duplicate leakage on an existing fork

```bash
python simple/near_duplicates.py \
  --data prepared/8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68 \
  --original release/simple-run \
  --fork release/unlearning-3i-atlas \
  --entity "3I/ATLAS" \
  --alias "C/2025 N1" \
  --out-filter topic_filter.json
```

Optional flags:
- `--alias`: Explicit additional entity aliases or catalog designations to scan for.
- `--control-entities`: Space-separated rarity-matched control entities (defaults to `1I/'Oumuamua 2I/Borisov`).
- `--sample-check`: Path to a JSON file containing hand-checked sample audit results (sets `sample_miss_rate` and records `sample_file_sha256`).
- `--out-filter`: Path to save the validated `topic_filter.json` specification artifact.

Prints a JSON summary with mention counts, control comparisons, bootstrap confidence intervals, and
token attributions, and optionally writes the validated `topic_filter.json` specification.
