# Privately reported finding: a forged record passed `verify`

Reported privately by dhruvi-16-me on 2026-10-28, as part of the [end-to-end evaluation](2026-10-28-end-to-end-evaluation.md).
Fixed in v1.0.2 (#267). Published after the fix was released.

## What she found

The CPU check in the README compared `chain.jsonl` with the chain head in `anchor.json`. Only the separate
`openssl ts -verify` step tied `anchor.json` to something outside the download. So someone who could change the
downloaded files could forge a record that `verify` accepted:

1. Edit the last entry of `chain.jsonl` so that its `model` digest points at different weights.
2. Recompute the chain head and write it into `anchor.json`.
3. Update `release-hashes.txt`. Nothing outside the download vouches for that file, so `sha256sum -c` would
   pass too.

`verify` then printed `chain head MATCH` and `MATCH chain entry 37` for the changed weights, and ended in PASS.
Only the timestamp check caught the forgery. The README listed that check as a separate step, which a reader
could skip.

She first worked this out from the code, then ran it: `verify` passed with modified weights, and both timestamp
checks failed on the forged `anchor.json`. She didn't rewrite `release-hashes.txt` in her run, but nothing would
have stopped it.

## Reproduction

On `main` before the fix (`4aee4a0`), with the released `conversation-model.pt`:

- one weight changed by 1e-3 and saved as a new file;
- the last chain entry's `model` set to the digest of that file;
- the chain head recomputed into `anchor.json`; the two `.tsr` files left unchanged.

Result: `anchor anchor.json: chain head MATCH (38 entries)`, `weights forged-model.pt: MATCH chain entry 37`,
`verify PASS`.

## Fix

Since v1.0.2, `verify` checks both RFC 3161 timestamps (`anchor.digicert.tsr`, `anchor.freetsa.tsr`) itself,
against the same `anchor.json` bytes that it compares with the chain head. It fails if a timestamp is missing or
doesn't verify. `--no-timestamps` skips the check and prints a warning, for runs that aren't anchored yet.

The same forged copy with v1.0.2:

```text
anchor anchor.json: chain head MATCH (38 entries)
timestamp anchor.freetsa.tsr does not verify for anchor.json: the anchor was changed after it was stamped
```

`verify` exits with status 1. The tests in `tests/simple/test_ovl_anchor.py` cover an edited anchor, a missing
timestamp, a reply that doesn't verify, a swap of `anchor.json` between the two checks, missing `openssl` and a
failed certificate download.

## What the timestamps prove

The timestamps prove that this exact `anchor.json` existed by 2026-10-20 14:38:43 UTC. A forger can't make a
new anchor look that old. The timestamps don't prove how the training was done; that takes recomputing segments
on an RTX 5090.
