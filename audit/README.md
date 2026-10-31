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
| Private | Reported privately | Fixed in #267. Details are added here after v1.0.2 is released |
| M1 | A missing `anchor.json` is skipped silently | Fixed in v1.0.1 (#252): it fails. `--no-anchor` skips it with a warning |
| M2 | The CPU check doesn't run the timestamp check | Fixed: the README includes the step (#253), and `verify` checks both timestamps itself (#267) |
| M3 | Terms used before they're defined | Open: docs rewrite |
| M4 | `tests/simple` needs an undocumented install step | Open: the legacy cleanup removes `tests/conftest.py` and `scripts/`; the docs rewrite gives one install command |
| L1 | The `openssl` CA path is Linux-only | Open: docs rewrite adds the macOS command. `verify` already finds `/etc/ssl/cert.pem` (#267) |
| L2 | The repository root is crowded | Open: the legacy cleanup leaves `simple/`, `tests/` and the docs |
| L3 | `verify` writes into the release folder | Open: docs rewrite says so |
| L4 | The recipe hash is never checked | Fixed in #267: a different `RECIPE` fails before any replay |
| L5 | The data check covers only `stream.json` | Open: the message will say exactly what it checks. Changed tokens are still caught at the segment replay |
| L6 | The lock file needs Python 3.12 | Open: docs rewrite states it; the legacy cleanup sets `requires-python = ">=3.12"` |
| L7 | A non-anchor file gives a `KeyError` | Fixed in #267: "not an anchor file" |
| L8 | The anchor result isn't recorded | Fixed in #267: recorded as a check, with its timestamps |
