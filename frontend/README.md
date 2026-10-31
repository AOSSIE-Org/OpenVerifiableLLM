# OpenVerifiableLLM Frontend (Site v3)

Static single-page project site for OpenVerifiableLLM, presenting real bit-for-bit training verification evidence from Hugging Face revision `96dc282a`.

## Overview

- **Single-page navigation**: Evidence-first landing page with section anchors (`#check`, `#verification`, `#timeline`, `#exclusion`, `#downloads`).
- **Real evidence data**: Pinned Hugging Face dataset `AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence` at revision `96dc282a`.
- **Prebuild validation**: `npm run build` runs `prebuild` (`npm run validate`), failing closed if any data contract or hash link is violated.
- **Accessible design**: WCAG 2.2 AA compliant, zero axe violations across themes (dark default, light opt-in) and viewports.
- **Offline bundle**: Self-hosted Geist fonts, embedded SVG brand assets, and zero runtime external CDN dependencies.

## Commands

```bash
cd frontend
npm ci
npm run dev        # start local development server at http://127.0.0.1:5173
npm run typecheck  # verify TypeScript types (tsc -b --noEmit)
npm test           # run Vitest suite (validation, copy lint, components)
npm run build      # validate evidence and build static output to dist/
npm run preview    # preview production build locally
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
│   ├── components/  # React components (Hero, Overview, CheckLadder, Verification, TrainingTimeline, etc.)
│   ├── content/     # copy.ts (single source of truth for text and copy linting)
│   ├── data/        # contracts.ts and validate.ts (strict schemas, invariant validator)
│   └── styles/      # tokens.css (color tokens and ratios) and base.css (typography and layouts)
└── tests/
    ├── browser/     # Playwright + axe-core accessibility and interaction test suite
    ├── fixtures/    # Corrupted evidence fixtures for fail-closed validation tests
    ├── components.test.tsx
    ├── copy-lint.test.ts
    └── validate.test.ts
```
