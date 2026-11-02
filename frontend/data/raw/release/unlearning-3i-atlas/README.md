# Verifiable unlearning: the simple-run model without the "3I/ATLAS" article

This folder is a **fork** of the published run in `release/simple-run/`. It is the same training,
except that one Wikipedia article was never trained on: **3I/ATLAS** (the interstellar comet;
page 80340464, revision 1371028825, 10,395 training targets). Every step can be checked, so the
claim "this model never saw that article" is verifiable rather than trusted.

## Why the claim holds

1. **Shared history, byte for byte.** Entries 0–30 of this `chain.jsonl` are identical to entries
   0–30 of `release/simple-run/chain.jsonl`. The article's first training target (6,915,731,634)
   comes after the cursor at boundary 30 (6,901,668,826), so no checkpoint up to 30 ever trained
   on it.
2. **A recorded fork.** Entry 31 (kind `fork`) has exactly the state of entry 30. Its control adds
   `exclude: {"wikipedia": [[6915731634, 10395]]}`, the article's exact target range. `verify`
   checks that a fork entry changes nothing except declaring the exclusion, and that the excluded
   range was not yet reached.
3. **Training without it.** From entry 31 on, the loader skips that document. Its tokens are
   never used as inputs or targets, because windows never cross documents. The per-step coverage
   check accepts a gap in target ids **only** at the recorded range, so any other missing or
   repeated data would stop training.
4. **Checked.** Every fork segment (30→31 through 37→38) was recomputed from its starting
   checkpoint and matched bit for bit, including the retrained last Wikipedia segment and the
   whole conversation phase. The fork's `conversation-model.pt` matches its final entry. See
   `verifications.jsonl`. They ran on the pod that trained the fork. The fork entry and the
   retrained Wikipedia segment (30→31, 31→32) were also recomputed on a separate, fresh RTX 5090
   using only the public files: MATCH. Anyone can repeat them (below).

## What removing one article changed (leave-one-out, `loo-results.json`)

| | Original (trained on 3I/ATLAS) | This fork (never saw it) |
|---|---:|---:|
| Loss on the 3I/ATLAS article's own text | 3.3604 | 3.3860 (+0.0256) |
| Mean loss on 50 control articles from the same segment (both trained on them) | 2.8710 | 2.8712 (+0.0002) |

Removing the article raised the loss on its own text about 100 times more than on articles both
models saw. That is the exact, causal effect of training on that one article, with no
approximation. A 23M model that sees each article once memorises little, and its generated text
about 3I/ATLAS is wrong either way. The point is that the effect can be measured and verified,
not that this small model knew the article.

## Files

- `chain.jsonl`, `recipe.json`: the fork's record (0–30 shared, 31 = fork, 32–38 new).
- `ckpt-00030.pt`: identical bytes to the original release's checkpoint 30.
  `ckpt-00031.pt`–`ckpt-00038.pt`: the fork's checkpoints.
- `base-model.pt`, `conversation-model.pt`: the models that never saw the article.
- `verifications.jsonl`, `loo-results.json`, `release-hashes.txt`.
- `code/`: the exact code. `simple/ovl.py` adds the `fork` command; with no exclusion it trains
  identically to the released code (checked on a real-data slice: identical chains).
  `simple/loo_eval.py` produces `loo-results.json`.

## Check it yourself (RTX 5090, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)

```bash
# with the prepared data from prepared/8de1d600…/ and this folder as RUN:
python code/simple/ovl.py verify --data PREP --run RUN --segment 30 31 32 33 34 35 36 37 \
    --weights RUN/conversation-model.pt
```
Segment 31→32 (the retrained Wikipedia segment) takes about 18 minutes; the rest take seconds.
