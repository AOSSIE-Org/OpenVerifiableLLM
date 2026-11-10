# Near-duplicate and topic exclusion

The exclusion fork (`release/unlearning-3i-atlas`) removed the article "3I/ATLAS" as one document.
Because other articles that mention the comet remained in the training stream, the model could still
learn facts about it from those secondary mentions. This page describes how secondary mentions are
detected, how residual loss is measured against rarity-matched controls, and how topic-level
exclusions are verified in the chain.

- [Why document exclusion leaves residual facts](#why-document-exclusion-leaves-residual-facts)
- [Scanning secondary mentions](#scanning-secondary-mentions)
- [Rarity-matched controls](#rarity-matched-controls)
- [Passage loss and token attribution](#passage-loss-and-token-attribution)
- [Bootstrap significance test](#bootstrap-significance-test)
- [Topic exclusion in the chain](#topic-exclusion-in-the-chain)
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
4. **Document filtering.** The primary excluded document is filtered out so only secondary mentions
   are evaluated.

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

## Topic exclusion in the chain

To remove every mention across the corpus rather than just a single document, `near_duplicates.py`
generates a topic filter specification (`topic_filter.json`):

```json
{
  "filter_name": "topic_near_duplicate_filter",
  "version": "1.0.0",
  "filter_code_sha256": "4a7b...",
  "target_entity": "3I/ATLAS",
  "matching_patterns": ["3I/ATLAS", " 3I/ATLAS", "C/2019 Q4"],
  "excluded_ranges_count": 14,
  "total_excluded_targets": 7168,
  "ranges": [[10523400, 512], [14982100, 512]],
  "sample_miss_rate": {"checked": 20, "misses": 0, "false_positives": 0}
}
```

The filter records:
- The SHA-256 digest of the extraction code.
- Hand-checked validation sample accuracy (misses and false positives).
- The exact target token ranges to exclude.

`simple/ovl.py fork` accepts `--exclude-topic-filter FILE`:
- Records the filter metadata and SHA-256 directly in `chain.jsonl` under `entry["fork"]["topic_filter"]`.
- Inserts the target ranges into `control["exclude"]["wikipedia"]`.
- `ovl.py train` and `ovl.py verify` then enforce exact coverage gaps on those ranges without
  modifying the training loop.

## Usage

### 1. Evaluate near-duplicate leakage on an existing fork

```bash
python simple/near_duplicates.py \
  --data prepared/8de1d600... \
  --original release/simple-run \
  --fork release/unlearning-3i-atlas \
  --entity "3I/ATLAS" \
  --out-filter topic_filter.json
```

Prints a JSON summary with mention counts, control comparisons, bootstrap confidence intervals, and
token attributions.

### 2. Fork by topic

```bash
python simple/ovl.py fork \
  --run release/simple-run \
  --data prepared/8de1d600... \
  --at 30 \
  --exclude-topic-filter topic_filter.json \
  --out run-fork-topic
```

Creates a new fork directory with `topic_filter` recorded in `chain.jsonl` and all matching ranges
excluded from checkpoint 30 onward.
