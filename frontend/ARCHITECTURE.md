# Frontend architecture

Maps `FRONTEND_CONTRIBUTOR_BRIEF.md` onto a static app we can merge.
Visual language lives in `DESIGN.md`. This file is structure, data, and
what “crazy” is allowed to mean.

---

## 1. Research: how AOSSIE actually ships frontends

What mentors already reward in this org (GSoC 2025–2026):

| Project | What it is | Frontend pattern | Lesson for us |
| --- | --- | --- | --- |
| **OrgExplorer** | GitHub org analytics | Vite + React + TS + Tailwind, **no backend**, GitHub Pages, IndexedDB cache | Sunny/less-cloud. Browser is the runtime. |
| **PictoPy** | Desktop gallery | React + Vite + Tailwind + ShadCN *inside Tauri*; separate static landing on `pictopy.aossie.org` | App UI ≠ marketing site. Landing is static Pages. |
| **PictoPy-Website / Resonate-Website / DebateAI-LandingPage** | Project landings | Vite/React, GH Pages, `*.aossie.org` | Org wants static landings, not Vercel. |
| **AOSSIE Website** | Org home | Next.js 16 + Tailwind v4 | Exception: the org site. **Our brief forbids a backend.** Do not copy Next. |
| **Devr.AI** | DevRel agent | React dashboard + FastAPI | Cloud-heavy. Opposite of this assignment. |
| **LandingPages idea (GSoC 2026)** | Bruno/M4dhav | Static pages, SEO, **better than generic AI landings** | Mentors have said out loud they can smell AI templates. |
| **UI idea** | Same mentors | Clean, minimal, **easy to maintain next year** | No throwaway Aceternity theme. |

Shared stack we should match: **React + TypeScript + Vite + static host**.
Shared philosophy: no paid APIs, no login required, GitHub Pages.

What they do **not** have: a site whose job is *refusing to overclaim*.
That is our opening. OrgExplorer graphs GitHub. PictoPy shows photos.
We show **a chain of checks that can be broken**. If we do that with
restraint, it is the most distinctive frontend in the org this year.

---

## 2. Product, in one sentence

A **static evidence docket**: public snapshot in, honest pages out.
It never trains, never signs, never calls GPU, never pretends a model exists.

```
approved JSON snapshot  →  validate  →  view models  →  pages
        ↑                         ↓
   fixtures (dev)          fail closed (no mock fallback)
```

The training pipeline (`src/`, PR #101, RunPod) is a different program.
We only *display* what the maintainer allows.

---

## 3. Runtime architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser  (GitHub Pages / local Vite)                       │
│                                                             │
│  HashRouter  ─  no server rewrites needed                   │
│                                                             │
│  Layout                                                     │
│    nav · ClaimStrip · <Outlet/> · footer                    │
│                                                             │
│  SnapshotProvider                                           │
│    mode: fixture | public-snapshot                          │
│    load → parse → validate → freeze                         │
│    on error: ErrorView  (never swap in a happy fixture)     │
│                                                             │
│  Pages                                                      │
│    /           Overview                                     │
│    /evidence   list + filters                               │
│    /evidence/:id  detail + chain rail                       │
│    /releases   base + chat cards                            │
│    /verify     five profiles                                │
│    /try        inference shell                              │
│                                                             │
│  InferenceGateway                                           │
│    public build → DisabledAdapter                           │
│    ?fixture=1  → MockAdapter + sticky warning               │
└─────────────────────────────────────────────────────────────┘
```

No React Query polling. No IndexedDB of visitor chats. OrgExplorer uses
IndexedDB because it *fetches GitHub*. We ship the snapshot **in the
build** (`public/data/`). Optional later: fetch one allowlisted URL with
integrity, still fail-closed.

**HashRouter** is the static-safe strategy the brief asks for. Direct
load of `/#/evidence/xyz` works on GitHub Pages without a 404.html hack.

---

## 4. Directory (implement this, not a larger tree)

```
frontend/
  README.md
  DESIGN.md
  ARCHITECTURE.md
  package.json
  package-lock.json
  index.html
  vite.config.ts          # base: './' until maintainer sets Pages path
  tsconfig.json
  src/
    main.tsx
    App.tsx               # providers + routes
    layout/
      AppShell.tsx
      ClaimStrip.tsx
      StatusMark.tsx
    pages/
      OverviewPage.tsx
      EvidenceListPage.tsx
      EvidenceDetailPage.tsx
      ReleasesPage.tsx
      VerifyPage.tsx
      TryPage.tsx
      ErrorPage.tsx
    data/
      contracts.ts        # types from brief §7.2
      validate.ts         # runtime checks, fail closed
      adapters/
        snapshotAdapter.ts
      fixtures/           # labelled synthetic JSON (imported by tests too)
    inference/
      types.ts
      DisabledAdapter.ts
      MockAdapter.ts
    styles/
      tokens.css
      base.css
  public/
    data/
      public-snapshot.json    # mode labelled; may be partial
      README.md               # how to refresh + revision
  tests/
    validate.test.ts
    status.test.ts
    releases.test.ts
    adapter.test.ts
    navigation.test.ts        # small browser test for unreleased flow
```

Do not add Tailwind/ShadCN/Aceternity unless a later PR needs it.
Ordinary CSS + the token table in `DESIGN.md` is enough and is closer
to “maintainable next year” than a UI kit.

---

## 5. Data contract (brief §7, implemented)

One versioned snapshot is the only store.

```ts
type CheckResult = 'PASS' | 'FAIL' | 'NOT_RUN' | 'UNAVAILABLE' | 'UNSUPPORTED';
type EvidenceScope = 'fixture' | 'pilot' | 'production';
type DataMode = 'fixture' | 'public-snapshot';
type ReleaseAvailability = 'not-released' | 'available' | 'withdrawn';
```

`validate.ts` must:

- reject unknown schema versions as **data error**, not PASS
- map only known legacy aliases (`passed` → PASS, `pending` → NOT_RUN)
- never coerce `"ok"` / `"complete"` / `"true"` to PASS
- require: snapshot mode, generatedAt, source refs, checks[], releases[]
- require each PASS to have profile, scope, explanation, evidence refs
- mark missing mandatory fields UNAVAILABLE or throw `SnapshotError`

Adapters keep `url`, `revision`, `path`, `sha256` as they came.
Hashes are displayed as the source string. We do **not** JSON.parse and
re-stringify then call that “the same bytes.”

`locallyRecomputed` is **always false** in this static app. Showing a
publisher JSON is not a local replay.

Failed public fetch: `ErrorPage`. No silent fixture.

---

## 6. Pages ↔ brief (everything must land)

| Brief | Module | Behaviour that cannot be skipped |
| --- | --- | --- |
| §5.1 Overview | `OverviewPage` | Purpose, **not released**, 3 actions, vertical pipeline with real statuses, provenance ≠ truth. No % complete, no fake date, no hero chat. |
| §5.2 Evidence | `EvidenceListPage` | Table/list primary. Filters: phase, kind, result, scope. Size + link, never auto-download dumps. |
| §5.2 Detail | `EvidenceDetailPage` | Digest copy, parents/children, missing parent as gap, superseded kept, bounded escaped JSON. |
| §5.3 Releases | `ReleasesPage` | Two cards. Disabled download/generate until metadata. No invented HF names. |
| §5.4 Verify | `VerifyPage` | Five profiles, distinct meanings. CLI = “Instructions pending release” until ryoari pastes a tested command. |
| §5.5 Try | `TryPage` + adapters | Public: generation unavailable. Fixture: labelled mock. Reject identity mismatch. No logging. |
| §6 Status | `StatusMark` | PASS/FAIL/NOT_RUN/UNAVAILABLE/UNSUPPORTED + text, not color-only. Empty checks ≠ success. |
| §8 Fixtures | `src/data/fixtures` | All listed cases exist as JSON; tests load them. |
| §9 a11y | CSS + tests | Keyboard, 200% zoom, reduced motion, contrast, announced copy. |
| §11 checklist | PR template in README | Tick with screenshots, not vibes. |

---

## 7. Inference boundary (brief §7.3)

```ts
interface InferenceAdapter {
  capabilities(): Promise<InferenceCapabilities>;
  generate(request: GenerationRequest, signal?: AbortSignal): Promise<GenerationResponse>;
}
```

- `DisabledAdapter`: capabilities.available = false. `generate` throws
  `GenerationUnavailable`.
- `MockAdapter`: capabilities.mock = true. Every response includes
  `disclaimer: 'Example UI response — not generated by a released OpenVerifiableLLM model'`.
  If `response.releaseId !== request.releaseId` → reject.
- Production bundle must not import a path that enables mock by default.
  Gate with `import.meta.env.VITE_INFERENCE=mock` **and** a visible UI flag.

No Ollama in v1. Connecting Ollama to unverified weights is how we
accidentally ship the product the brief forbids.

---

## 8. What can be “crazy” (attention) vs what cannot

Mentors (Bruno, Archit, ryoari) punish fake green and template landings.
They notice **one true interaction** that the rest of AOSSIE does not have.

### Build these — they are distinctive and on-brief

1. **Claim strip (signature)**  
   Every route states what the page may and may not claim, plus snapshot
   time and source commit. Screenshot this in the PR.

2. **Broken chain rail**  
   Parent → this → children. Missing parent is a physical gap. This is
   the G09 “publication must not overclaim” idea as UI.

3. **Fail-closed empty docket**  
   A designed screen for “zero checks” and “malformed snapshot.” Most
   dashboards fake a happy empty state. Ours must look unfinished.

4. **Supersession timeline**  
   FAIL report remains, struck, with an arrow to the later PASS. History
   is not rewritten. Unique vs OrgExplorer’s live graphs.

5. **Narrow browser checksum (optional M3+)**  
   User drops a *small* published file; Web Crypto SHA-256 compared to
   the snapshot digest. Label: **checksum of this file in this browser,
   not training replay, not publisher trust.** Fits “browser checksum
   must be labelled as that narrow check.”

6. **Fixture / public mode switch that is ugly on purpose**  
   A persistent “FIXTURE” tape on the viewport when not in public mode.
   Impossible to screenshot by accident as production.

### Do not build — looks busy, costs GSoC trust

- Magic UI / Aceternity / 3D / custom cursor / glass hero  
- RAG “this sentence is from Wikipedia page N” (the txt diagram)  
- Live Ollama chat as the homepage  
- Training/RunPod/cost dashboards  
- Invented CLI copy buttons  
- Accounts, wallets, visitor prompt logs  

Cinematic kits stay as **140–220ms opacity/transform** on copy, expand,
and pipeline stagger. That is the synthesis of the reference txt.

---

## 9. Tests (brief §11)

Run offline. No Wikipedia dump, no GPU.

| File | Asserts |
| --- | --- |
| `validate.test.ts` | bad schema → error; empty checks → not success |
| `status.test.ts` | alias map; `"ok"` is not PASS |
| `releases.test.ts` | not-released hides download href |
| `adapter.test.ts` | mock identity mismatch rejected; disabled throws |
| `navigation.test.ts` | Playwright/Vitest browser: overview → evidence → unreleased releases |

CI later: `frontend.yml` only, on `frontend/**`. Do not edit signing
or training workflows.

---

## 10. Git rules for this repo

- Branch from **current `main`**: `frontend/evidence-docket`
- Touch **only** `frontend/`
- Never commit `keys/`, `.venv`, `tmp_smoke`, `src/` LSTM hacks, `goal_state.json`
- Do not force-push shared branches
- Draft PR as soon as `npm run dev` shows overview + claim strip
- Small commits, one concern each (scaffold / types / pages / tests)

Handoff in the PR body: screenshots mobile+desktop, fixture vs public,
missing parent, unreleased releases, exact maintainer inputs still needed.
