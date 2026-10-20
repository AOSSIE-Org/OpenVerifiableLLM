# OpenVerifiableLLM frontend

A static site for the OpenVerifiableLLM public record: evidence, releases,
verification profiles and an inference page. It displays a saved JSON snapshot.
It has no backend and does not train, sign or run a model. The project is at
an early stage and no model has been released yet.

## Requirements

Node 20.19+ or 22.12+ (required by Vite 7) and npm.

## Commands

```sh
cd frontend
npm ci
npm run dev        # http://localhost:5173
npm run typecheck
npm test
npm run build      # writes dist/
npm run preview    # serves dist/
```

Routing uses `HashRouter` (`/#/evidence`) and `vite.config.ts` sets
`base: './'`, so the build can be served from any static host or sub-path.

## Data

Snapshots live in `src/data/fixtures/`. The app loads one, runs it through
`validateSnapshot` in `src/data/validate.ts`, and shows a data error page if it
is invalid rather than an empty or passing state. Choose a pack with
`?snapshot=`, for example `/#/?snapshot=missing-parent`:

| `?snapshot=` | File | Shows |
| --- | --- | --- |
| `public` (default) | `public-snapshot.json` | Draft project status: early practice-run checks passed, both models not released, replay not run |
| `missing-parent` | `missing-parent.json` | An evidence item whose parent is not in the record |
| `superseded` | `superseded.json` | A failed check kept after a later pass |
| `empty` | `empty-checks.json` | No checks, shown as a data error |

All of these are fixtures (`"mode": "fixture"`) and the site shows a banner
while one is loaded. The default pack is a draft of the current project status
and has not been approved by the maintainer. `public/data/` is reserved for the
approved snapshot, which is the only file that should use
`"mode": "public-snapshot"`.

## Structure

- `src/data/contracts.ts`: snapshot types.
- `src/data/adapters/snapshotAdapter.ts`: loads the selected fixture.
- `src/data/validate.ts`: checks the snapshot and fails closed.
- `src/snapshot.tsx`: provides the validated snapshot to the pages.
- `src/pages/`: Overview, Evidence, Releases, Verify and Inference.
- `src/content/copy.ts`: shared wording for pipeline stages and the FAQ.
- `src/inference/`: `DisabledAdapter` and `MockAdapter`. Neither is used by
  the app yet; the Inference page is a disabled placeholder and `MockAdapter`
  is only used in tests.
- `tests/`: Vitest tests for validation, fixture loading, release gating,
  stage results and the inference adapters.

Everything is in `frontend/`. Nothing here touches the pipeline in `src/`,
`project/` or signing keys.

## Not done yet

- Real project data, once the maintainer approves a public snapshot (M3).
- An inference preview using `MockAdapter` behind a build flag (M4).
- Copyable verification commands. The Verify page says instructions are
  pending until a tested command exists.
- Browser tests, a frontend CI workflow and deployment (M5).

## Needed from the maintainer

- An approved public snapshot and its source revision.
- A tested verification command for each profile.
- Release metadata for the base and chat models when they exist.
- The hosting target and base path.
