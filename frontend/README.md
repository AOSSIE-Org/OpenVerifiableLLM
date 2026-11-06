# OpenVerifiableLLM Frontend (Site v3)

Static single-page project site for OpenVerifiableLLM, presenting real bit-for-bit training verification evidence from Hugging Face revision `96dc282a`.

## Overview

- **Single-page navigation**: One continuous report with section anchors (`#how`, `#overview`, `#timeline`, `#check`, `#downloads`, `#faq`).
- **Pinned evidence**: `public/data/evidence.json` is a committed snapshot of Hugging Face revision `96dc282a` and the `v1.1.0` GitHub release. The dataset's `main` branch has moved on, so links and commands use the revision, not `main`.
- **Results come from the records**: Every result, count, date and revision on the page is worked out from the snapshot (`src/data/facts.ts`, `src/content/claims.ts`). A mismatch shows as a mismatch, and a missing check shows as not checked. The positions of the host failure and the code change are written by hand from `docs/how-it-works.md`, and a test keeps them in line with the evidence.
- **Accessible design**: Targets WCAG 2.2 AA, with axe run across both themes (light default, dark opt-in) and several widths.
- **Offline bundle**: System fonts only (no web fonts), embedded SVG brand assets, and zero runtime external CDN dependencies.

## What the validator does and doesn't check

`src/data/validate.ts` runs before every build and on every page load. If a check fails, the build stops and the page shows an error instead of the evidence.

It checks that the snapshot is consistent with itself:

- **Chain**: each entry's hash is recomputed from its published record, using the same bytes `ovl.py` hashed. Each entry links to the one before, and the displayed fields must equal the record.
- **Anchor**: the records must rebuild the `chain.jsonl` the anchor names, and the anchor's head, final state and final model must match the last entry.
- **Checks**: coverage and counts must equal what the raw rows say, and every checker must be listed in `src/data/checkers.json`.
- **Fork**: the split must link to the run's own entry and keep its state.
- **Values used in links and commands**: the dataset, release tag and revision must match their exact formats.

**This is not cryptographic verification.** It finds accidental damage and partial edits. It can't tell who wrote the snapshot: someone who rewrites the records, every hash and the anchor together would pass, and a test records this. The validator doesn't check the timestamp signatures, doesn't compare check rows or the hardware map with any signed record, and doesn't recompute training. To verify the evidence, run the checks on the page: `simple/ovl.py verify` checks the timestamps, and the segment check recomputes training.

## Commands

Run these from `frontend/`:

```bash
npm ci
npm run dev             # development server at http://localhost:5173
npm run typecheck       # TypeScript
npm test                # unit, validator and copy tests
npm run validate        # validate the committed snapshot only
npm run build           # validate the snapshot, then build to dist/
npm run preview         # serve the production build
npm run fetch-evidence  # rebuild the snapshot from the pinned sources (downloads the release assets)
```

Browser tests serve `dist/`, so build first or they test an old bundle:

```bash
npx playwright install chromium
npm run build
npm run e2e             # PW_CHANNEL=chrome uses an installed Chrome instead
```

## Structure

```
frontend/
├── public/
│   ├── brand/       # Segment check SVGs (logo-dark, logo-light, favicon, app-icon, aossie-logo)
│   └── data/        # evidence.json (pinned data committed for offline validation)
├── scripts/
│   └── fetch-evidence.mjs  # Manual script to fetch and assemble snapshot from Hugging Face
├── src/
│   ├── components/  # React components (Hero, HowItWorks, Results, TrainingHistory, CheckLadder, etc.)
│   ├── content/     # copy.ts (static text) and claims.ts (sentences worked out from the evidence)
│   ├── data/        # contracts.ts, facts.ts (results and counts), validate.ts, sha256.ts, checkers.json
│   └── styles/      # tokens.css (color tokens and ratios) and base.css (typography and layouts)
└── tests/
    ├── browser/     # Playwright + axe-core accessibility and interaction test suite
    ├── evidence-cases.ts  # Mismatches and missing checks, built from the real snapshot
    ├── components.test.tsx
    ├── facts.test.ts
    ├── copy-lint.test.ts
    └── validate.test.ts
```
