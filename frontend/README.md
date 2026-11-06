# OpenVerifiableLLM Frontend (Site v3)

Static single-page project site for OpenVerifiableLLM, presenting real bit-for-bit training verification evidence from Hugging Face revision `96dc282a`.

## Overview

- **Single-page navigation**: One continuous report with section anchors (`#how`, `#overview`, `#timeline`, `#check`, `#downloads`, `#faq`).
- **Real evidence data**: Pinned Hugging Face dataset `AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence` at revision `96dc282a`.
- **Results come from the records**: Every result, count, date and revision on the page is worked out from `evidence.json` (`src/data/facts.ts`, `src/content/claims.ts`). A mismatch shows as a mismatch, and a missing check shows as not checked.
- **Prebuild validation**: `npm run build` runs `prebuild` (`npm run validate`), failing closed if the snapshot disagrees with itself: chain links, the anchor, the hash list, or a summary that differs from the raw check rows.
- **Accessible design**: Targets WCAG 2.2 AA, with axe run across both themes (light default, dark opt-in) and five widths.
- **Offline bundle**: System fonts only (no web fonts), embedded SVG brand assets, and zero runtime external CDN dependencies.

## Commands

```bash
cd frontend
npm ci
npm run dev        # start local development server at http://127.0.0.1:5173
npm run typecheck  # verify TypeScript types (tsc -b --noEmit)
npm test           # run Vitest suite (validation, copy lint, components)
npm run build      # validate evidence and build static output to dist/
npm run preview    # preview production build locally
npx playwright install chromium
npm run e2e        # browser and accessibility tests (PW_CHANNEL=chrome uses an installed Chrome)
npm run fetch-evidence  # rebuild evidence.json from the pinned sources
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
│   ├── data/        # contracts.ts, facts.ts (results and counts), validate.ts (invariants), checkers.json
│   └── styles/      # tokens.css (color tokens and ratios) and base.css (typography and layouts)
└── tests/
    ├── browser/     # Playwright + axe-core accessibility and interaction test suite
    ├── evidence-cases.ts  # Mismatches and missing checks, built from the real snapshot
    ├── components.test.tsx
    ├── facts.test.ts
    ├── copy-lint.test.ts
    └── validate.test.ts
```
