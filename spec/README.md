# Lean spec

Machine-checked statements of what `ovl.py verify` establishes. Lean 4.35.0, no other dependencies.

| File | What it proves |
|---|---|
| `OVL/Chain.lean` | Two chains that pass the link check and match the same anchor are the same chain, or they contain a SHA-256 collision (`anchor_binds_chain`). |
| `OVL/Odds.lean` | With N segments, M random independent checks miss one wrong segment with probability ((N−1)/N)^M (`count_miss`). For N = 37: 110 checks give 95% and 169 give 99%, and 109 and 168 are not enough (`odds_95`, `odds_99`). |
| `OVL/Exclusion.lean` | For ranges sorted by start, `next_target` skips exactly the excluded targets (`le_nextTarget`, `skipped_excluded`, `nextTarget_not_excluded`). |

```bash
cd spec
lake build                    # checks every proof
lake env lean Axioms.lean     # lists the axioms each theorem uses
```

CI fails if a theorem uses `sorry` or any axiom besides Lean's standard three. Add every new
theorem to `Axioms.lean`.

## What these proofs don't cover

- They are about models of the Python code, not the code itself. The models are short and follow
  `simple/ovl.py` line by line; `tests/simple/test_spec_exclusion.py` checks the Python
  `next_target` against the exclusion theorems.
- `anchor_binds_chain` assumes that canonical JSON (`json.dumps(..., sort_keys=True)`) gives
  different bytes for different entries.
- Nothing here proves that a GPU computed a training step correctly. That is what recomputing a
  segment bit for bit checks.
- The odds hold only if each checker picks a segment at random, independently of the trainer.
