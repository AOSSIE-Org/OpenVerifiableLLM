# OpenVerifiableLLM evidence site (v3)

A static site that shows what was trained, who has checked it, and how to check it yourself.
Every number comes from the Hugging Face dataset
`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence` pinned at `96dc282a`, and every
sentence comes from the claim ledger. Nothing runs on a server.

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
npm run claims:sync     # CLAIMS_DIR=/path/to/ovllm-claims; copies claims.json and COPY.md
npm run data            # rebuild public/data/site-data.json from data/raw
npm run lint && npm run typecheck && npm test && npm run contrast
npm run build           # data drift check → typecheck → client + SSR build → prerender → phone budget
npx playwright install chromium && npm run test:e2e
```

`npm run build` fails if `site-data.json` differs from what the pinned files produce, if any
section fails validation, if a ledger claim has no source, uses a banned phrase, or shows a
number found neither in its cited lines nor in `site-data.json`, or if the bundle exceeds the
phone budget. Set `BASE_PATH=/OpenVerifiableLLM/` to build for a sub-path.
