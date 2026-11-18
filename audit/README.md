# Audits

Outside reviews of the project, and what we did about each finding.

| Date | Reviewer | Scope | Report |
|---|---|---|---|
| 2026-10-28 | [dhruvi-16-me](https://github.com/dhruvi-16-me) | End to end on macOS (CPU): docs, the CPU check, timestamps, tampered copies, tests | [report](2026-10-28-end-to-end-evaluation.md) |

## Responses to the 2026-10-28 evaluation

The review covered commit `748280a` (the v1.0.0 code). v1.0.1 (28 Oct) fixed M1 and M2 before the
report arrived.

| ID | Finding | Status |
|---|---|---|
| H1 | The released `base-model.pt` fails `verify` | Fixed in #267: any `base` or `final` match is accepted. `base-model.pt` gives `MATCH chain entry 31` |
| Private | A forged record passed `verify` | Fixed in v1.0.2 (#267): `verify` checks both anchor timestamps itself. [Write-up](2026-10-28-private-finding.md) |
| M1 | A missing `anchor.json` is skipped silently | Fixed in v1.0.1 (#252): it fails. `--no-anchor` skips it with a warning |
| M2 | The CPU check doesn't run the timestamp check | Fixed: the README includes the step (#253), and `verify` checks both timestamps itself (#267) |
| M3 | Terms used before they're defined | Fixed in the docs rewrite: the README defines them in "Words used here" before using them |
| M4 | `tests/simple` needs an undocumented install step | Fixed: the cleanup (#271) removed `tests/conftest.py` and `scripts/`, and CONTRIBUTING gives one setup command |
| L1 | The `openssl` CA path is Linux-only | Fixed: `verify` finds `/etc/ssl/cert.pem` (#267), and `docs/verify.md` gives the macOS command |
| L2 | The repository root is crowded | Fixed in #271: the legacy code is removed, and all the code is in `simple/` |
| L3 | `verify` writes into the release folder | Documented in `docs/verify.md` |
| L4 | The recipe hash is never checked | Fixed in #267: a different `RECIPE` fails before any replay |
| L5 | The data check covers only `stream.json` | Fixed: the message names the `stream.json` hashes, and `docs/verify.md` explains that changed tokens are caught at the segment replay |
| L6 | The lock file needs Python 3.12 | Fixed: CONTRIBUTING states it, and `pyproject.toml` requires 3.12 (#271) |
| L7 | A non-anchor file gives a `KeyError` | Fixed in #267: "not an anchor file" |
| L8 | The anchor result isn't recorded | Fixed in #267: recorded as a check, with its timestamps |
