# OpenVerifiableLLM frontend

A static evidence page for OpenVerifiableLLM, laid out like a Wikipedia article: a contents list, an
infobox, numbered sections, plain tables and numbered references to the pinned evidence files. No
3D, no scroll effects, no animation.

Every number comes from `public/data/evidence.json`, built from the Hugging Face dataset
`AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence` at revision `96dc282a`, and is
validated on load. If the snapshot is missing or fails validation, the page shows an error and no
numbers.

## Sections

1. Overview: headline facts and limits
2. Check it yourself: try it, consistency check, actual verification
3. Verification: coverage map (project / independent), detection odds, hardware map, all checks
4. Training timeline: the 38 chain boundaries, host restart (segment 10), code change (from 19)
5. Verifiable exclusion: the 3I/ATLAS fork, loss numbers, why the claim holds, limits
6. Downloads: each file with its SHA-256
7. Terms used on this page
8. References: each evidence file, linked at the pinned revision, with its SHA-256

## Commands

```bash
cd frontend
npm ci
npm run fetch-evidence  # rebuild public/data/evidence.json from Hugging Face
npm run dev             # http://127.0.0.1:5173
npm run typecheck
npm test                # validator, copy lint, no-motion check, components
npm run build           # validates the snapshot first, then builds dist/
npx playwright test     # browser and axe accessibility checks (light and dark, 3 widths)
```

## Structure

```
frontend/
├── public/data/evidence.json   # pinned snapshot
├── scripts/fetch-evidence.mjs  # downloads the files at the pinned revision and writes the snapshot
├── src/
│   ├── components/             # Header, Contents, Infobox, sections, References, Footer
│   ├── content/copy.ts         # all page text
│   ├── data/                   # contracts.ts, validate.ts
│   └── styles/base.css         # the only stylesheet
└── tests/                      # vitest, fixtures, Playwright
```
