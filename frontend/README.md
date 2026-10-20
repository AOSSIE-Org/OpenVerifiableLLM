# OpenVerifiableLLM frontend

An exportable, static React/TypeScript frontend implementing the supplied Figma design and the frontend contributor brief dated 2026-09-19. All changes are confined to this directory.

**The bundled snapshot is explicitly synthetic.** This deliverable is a frontend, not a training verifier, completed model release, or live inference service.

## Start

Use Node **22.12 or later** (tested with Node 24.19.0 and npm 11.9.0).

```sh
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. Development ordinarily uses port 5173. No API keys, account, database, model downloads, external font requests, or proprietary services are needed. Package dependencies are pinned exactly and package-lock.json includes integrity hashes. The three font families are self-hosted through Fontsource.

### Checks and static build

```sh
npm run typecheck
npm test
npm run build
npm run check:production
npx playwright install chromium
npm run test:browser
npm run preview
```

The browser suite serves the existing production build at `/OpenVerifiableLLM/` without an SPA fallback and separately starts a development mock server. Run `build` first. The browser suite needs Chromium and permission to start local servers. If a browser download mirror fails, Playwright tries alternate mirrors. An optional `PLAYWRIGHT_BROWSERS_PATH` may select an existing browser cache.

The included `dist/` is a prebuilt example using the bundled synthetic snapshot. Rebuild after changing code, environment configuration, or metadata. Host only `dist/`, not the source tree.

## Views and behavior

| View | Route | Behavior |
| --- | --- | --- |
| Overview | `#/` | Purpose, release availability, declared process, workflow disclosure, scoped evidence and snapshot provenance. |
| Evidence Explorer | `#/evidence` | Search title/ID/digest; phase/kind/result/scope filters; URL-preserved state; accessible records. |
| Evidence Detail | `#/evidence/<id>` | Named checks, full digest copy, source/revision, attribution, parents/children, supersession, bounded technical metadata. |
| Releases | `#/releases` | Separate base/chat roles; honest unavailable actions; future exact-identity rendering and revision-pinned files. |
| Verification | `#/verification` | Five distinct profiles, boundaries/prerequisites, inspected results, approved-command gating. |
| Inference | `#/inference` | Generation unavailable in every public production build. Optional isolated development-only mock. |

All pages adapt at narrow mobile widths. Evidence rows become compact records while retaining table headings and semantics. Long hashes and revisions wrap. Navigation has a keyboard-accessible mobile menu and skip link. Internal route changes focus the main region; initial page load preserves normal keyboard order.

Routes live in the URL hash, so a bookmarked evidence detail only requests the containing index.html from a static host. Filters are retained in hash-query parameters. There is no server-side rewrite requirement.

## Base path

The default Vite base is `./`. This supports any static directory:

```text
https://example.org/OpenVerifiableLLM/#/evidence/synthetic-pilot-replay
```

If the hosting owner confirms a fixed prefix, set `VITE_BASE_PATH=/OpenVerifiableLLM/` at build time. Always visit the deployment directory with a trailing slash. Keep `data/` beside `index.html`. The snapshot loader resolves metadata inside the same deployment directory and rejects external or escaping paths.

No host, deployment workflow, public URL, permissions, or secrets were changed.

## Metadata contract

`src/data/contracts.ts` is the authoritative version-1 presentation schema and runtime validator. It is a display contract, not a replacement for signed manifests, trust policy or pipeline verification schemas.

Required snapshot fields: `schemaVersion`, `generatedAt`, `mode`, `approval`, `sources`, `workflow`, `releases`, `checks`, `evidence`, `commands`. Required arrays never silently default to empty. Unknown critical versions and malformed data stop rendering the data.

- Verification: PASS / FAIL / NOT_RUN / UNAVAILABLE / UNSUPPORTED.
- Workflow: pending / running / complete.
- Browser data: loading / ready / stale / error.
- Scope: fixture / pilot / production.
- Data mode: fixture / public-snapshot.

These are distinct values. A loaded report never becomes recomputation, workflow completion never becomes a passing check, and zero checks never become success. Reported local recomputation is attributed to its supplied report; **this visitor is always shown as not having recomputed verification**.

Every evidence record has stable ID, human title, phase/kind/scope, synthetic designation, nullable source fields, parent IDs, size/time when known, supersession references and bounded flat technical metadata. Supporting check references are separate stable IDs. Missing references remain visible and navigable to the missing-evidence state. Original failed/superseded reports are preserved.

An available release requires exact repository, immutable revision, root, file inventory with sizes/digests, supported runtime and license/model-card references. Chat releases need a base parent. Download URLs are assembled from the exact revision and declared file paths. Synthetic future-release fixtures never enable downloads.

Sources can be absent explicitly; unknown source URLs, operators and dates are not fabricated. External links accept HTTPS only and disallow credentials. React escapes untrusted text and JSON. No raw HTML or Markdown rendering is used.

The loader fetches only a configured small display snapshot, stops after 1,000,000 bytes and does not fetch datasets, checkpoints, report payloads or model archives. Detail metadata display is bounded to 8,192 characters; expanding it does not fetch a large artifact.

### Fixtures and development scenarios

Default: `public/data/snapshot.json`, 10,127 bytes. Fixture IDs and operators are explicitly synthetic; context links to the project are not claimed as sources for synthetic results.

Copy `.env.example` to `.env.local` for local settings. To exercise a development scenario:

```dotenv
VITE_DATA_MODE=fixture
VITE_SCENARIO=empty
```

Supported scenarios in `src/data/scenarios.ts`: `empty`, `no-evidence`, `malformed`, `unsupported-schema`, `missing-evidence`, `stale`, `fetch-error`, `public-unapproved`, `future-release`. Other values use the ordinary synthetic fixture. Development scenarios are removed from production JavaScript.

The default scenario already includes scoped pilot PASS with production NOT_RUN, a failed integrity check, an unavailable reconstruction report, unsupported tokenizer reconstruction, a missing parent and a superseded failure. Browser tests inject actual malformed/network/empty response states. No simulated failure replaces data with successful fixtures.

### Approved public metadata, when supplied

No approved public snapshot was available for this assignment. There is intentionally no invented “approved” snapshot.

1. Have the maintainer choose a small pinned display snapshot and review its field mappings. The runtime adapter expects the versioned presentation contract; it does not guess legacy status semantics.
2. Preserve raw artifact URLs/revisions/digests and all reported check fields. If legacy values must be converted, implement and test a named adapter under `src/data/`. Never translate arbitrary truthy strings into PASS.
3. Record mode `public-snapshot`, explicit approval identity/time/immutable revision, and a source reference with the same pinned revision.
4. Put the reviewed snapshot under `public/data/` and set both `VITE_DATA_MODE=public-snapshot` and `VITE_SNAPSHOT_PATH=data/<reviewed-file>.json`.
5. Run the full frontend checks and rebuild. A mode mismatch, synthetic evidence in public mode, missing approval, malformed data or failed fetch renders an error. There is no fixture fallback.

The schema validates the shape and binding of approval metadata; it does **not authenticate maintainer approval** or verify a signature. The publishing maintainer must approve the actual content. Do not recursively publish project/cache/operational directories.

## Verification instructions

No final CLI syntax is supplied. All five profiles explain their limits and prerequisites. The default page says “Instructions pending release.” Commands appear only when approved=true, required metadata is complete, a source revision is present, resource estimates have a measurement source, and validation evidence exists in the snapshot. Executable copying is enabled only in public-snapshot mode.

Commands are displayed/copied as text, never run by the browser. Full replay is not an instant browser action. Resource metadata must distinguish measured usage from estimates; no cost figure is fabricated.

## Development inference and later integration

```dotenv
VITE_MOCK_INFERENCE=true
```

Restart the dev server, then visit `#/inference`. The development panel is imported only if `import.meta.env.DEV` and this build-time flag are both true. Production cannot enable it with any URL flag or UI toggle; it is tree-shaken out, and `check:production` checks the emitted JS.

The adapter interface in `src/inference/adapter.ts` defines capabilities, exact release identity, input/history, settings, bounded requests and responses. It checks selected and returned release identities and settings. Only greedy example decoding and 128 output tokens are supported. Input limit: 2,000 UTF-16 code units, including conversation history. Examples cover success/error/timeout/empty/identity mismatch; cancellation, retry, history clearing and copying are implemented.

Every mock response/history entry visibly says **Example UI response — not generated by a released OpenVerifiableLLM model.** All prompts/history stay in component memory; there is no analytics, prompt logging or persistent storage. Cancelling a browser request does not promise a future server stopped computation.

Before replacing the mock, the maintainer must supply an exact released-model identity, supported inference interface and capabilities/settings, resource limits, privacy policy and cancellation behavior. Do not silently substitute another model or external API. No receipt download or streaming claim is implemented without a real supported receipt/backend.

## Integration

Place this directory under the upstream repository as `frontend/` on your own branch. The supplied archive had no existing frontend. Preserve any frontend or unrelated edits added since that archive: compare before copying, merge consciously, and do not overwrite another contributor's changes.

Reference inputs inspected:

- Supplied Figma file: https://www.figma.com/design/nriGWR7KYdZfZwq5Uo4aL0
- Supplied contributor brief and attached implementation prompt.
- Supplied repository archive: CONTRIBUTING.md, docs/PIPELINE_FORMAT.md, docs/VERIFIABLE_WIKIPEDIA_PLAN.md (including sections 14–15).

`PROJECT_GOAL.md` and `project/goal_state.json` were not present in the supplied archive. No contents were fabricated and no real progress mapping was attempted. Public GitHub retrieval was unavailable. The frontend makes no live progress claim.

No training/verification/signing code, mutable progress state, Python dependencies or existing workflow was edited. No deployment or PR submission was performed.

See [CHECKLIST.md](CHECKLIST.md), [VALIDATION_REPORT.md](VALIDATION_REPORT.md), [PR_HANDOFF.md](PR_HANDOFF.md) and `screenshots/` for the review packet.

## License and AI assistance

When integrated upstream, contributions follow the repository's existing license. Font licenses are preserved under `licenses/`; the distributed font packages use the SIL Open Font License. Other dependency license records remain in their npm packages.

This frontend, fixtures, documentation and tests were produced with AI assistance using Codex and the Figma plugin. This disclosure does not claim human review or maintainer approval. The supplied CONTRIBUTING.md contains no separate AI-disclosure rule; recheck the current policy before opening a PR.

## Visual refresh notes

- Palette: `#f4efe6` paper, `#fefcf6` surface, `#162a2c` ink, `#5e6c5b` sage, `#d6e0e2` mist, `#686867` muted (CSS variables at the top of `src/styles.css`).
- Fonts: Bricolage Grotesque (headings) and Instrument Sans (body); IBM Plex Mono is kept only for digests and code.
- shadcn-style structure: `components.json`, `@/` alias, `src/lib/utils.ts`, `src/components/ui/`. Tailwind v4 loads theme and utilities only (no preflight) in `src/tailwind.css`, so the hand-written CSS is not reset.
- `container-scroll-animation.tsx` powers the Evidence page card; `globe.tsx` (cobe 0.6.3) sits in the shared footer in `src/chrome.tsx`.
