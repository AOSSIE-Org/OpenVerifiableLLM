# OpenVerifiableLLM evidence site (v3)

A static site that shows what was trained, who has checked it, and how to check it yourself.
Pinned evidence numbers come from the Hugging Face dataset
`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence` at `96dc282a`. Commands, cost, and
release digests come from GitHub copies listed below. Headlines come from `copy.json`; labels from
`ui.json`; facts and limits from the claim ledger. Nothing runs on a server.

## Where things come from

| What | File | Owner |
|---|---|---|
| Pinned evidence files + sha256 manifest | `data/raw/**` (`npm run data:fetch`) | fetched, never edited |
| GitHub docs cited by the ledger, at `1049d2a7` | `data/upstream/1049d2a7/**`, `data/upstream/manifest.json` | copied verbatim |
| GitHub release v1.0.1 asset digests | `data/upstream/github-release-v1.0.1.json` | API snapshot |
| Site data (every value has a source row) | `public/data/site-data.json` (`npm run data`) | generated |
| Claims: ids, sources, limits, banned phrases | `src/content/claims.json` | Research Lab (`npm run claims:sync`) |
| Plain-language headlines per claim id | `src/content/copy.json` (from `COPY.md`) | Open Source Ops (`npm run claims:sync`) |
| Navigation and table labels (no facts, no numbers) | `src/content/ui.json` | site |
| Ledger numbers the pinned data does not support | `src/content/ledger-issues.json` | reported, withheld from the page |

## Commands

```bash
npm ci
npm run data:fetch      # pinned evidence files into data/raw (needed by 3 unit tests and the build)
CLAIMS_DIR=/path/to/ovllm-claims npm run claims:sync   # copies claims.json and COPY.md
npm run data            # rebuild public/data/site-data.json from data/raw
npm run lint && npm run typecheck && npm test && npm run contrast
npm run build           # data drift check → typecheck → client + SSR build → prerender → phone budget
npx playwright install chromium && CI=1 E2E_PORT=4391 npm run test:e2e   # own port; never reuses a running server
```

`npm run data:check` (and therefore `npm run build` and CI) fails if the committed
`public/data/site-data.json` differs from a fresh rebuild of the pinned files and
`src/content/claims.json`. The build also fails if any section fails validation, if a ledger
claim has no source, uses a banned phrase, or shows a number found neither in its cited lines
nor in `site-data.json`, or if the bundle exceeds the phone budget. Set
`BASE_PATH=/OpenVerifiableLLM/` to build for a sub-path.

Hashed copies under `data/raw/`, `data/upstream/`, and `public/data/` are marked `-text` in
`.gitattributes` so a Windows `core.autocrlf` checkout cannot rewrite the bytes.

## How to change one claim

Edit that claim in `src/content/claims.json` (id, sources, limits, wording). If the headline
should change, edit the same id in `src/content/copy.json` (or in `COPY.md`, then
`npm run claims:sync`). Run `npm run data` so `public/data/site-data.json` is rebuilt. Leave
`ui.json` for labels that carry no facts and no numbers.

## Counts in copy

Headlines never type a segment or check count. `copy.json` uses placeholders filled from `site-data.json`:
`{matched}` `{total}` `{independent}` `{rows}` `{second}` `{training}` `{fresh}` `{m95}` `{m99}` `{N}` (OV-04, VE-01, VE-03, VE-07, VE-08). A line
that says "{matched} of {total}" or "all matched" is withheld (scoped error) when the data disagrees. The
ledger keeps its literal numbers; `tests/unit/fill.test.tsx` fails if a filled value is not in that claim's
ledger text. COPY.md must carry the same placeholders, or `npm run claims:sync` reverts them (the test fails).

## Data pinning

Each prerendered page carries `<meta name="ovl-data">` (sha256 of the data it was built from) and loads
`data/site-data.<sha16>.json`. Data with another hash never hydrates the page: every data section shows
"Couldn't load this evidence." A CSP meta tag ships on every page (no inline scripts).
