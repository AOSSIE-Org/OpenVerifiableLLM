# OpenVerifiableLLM frontend

Static evidence docket. It does **not** train, sign, or talk to a live model.

Requires Node 20+.

```sh
cd frontend
npm install
npm run dev
npm run test
npm run typecheck
npm run build
```

Open http://localhost:5173 — HashRouter, so routes look like `/#/evidence`.

## Snapshot packs

The app never scrapes `goal_state.json`. It loads a frozen JSON pack:

| `?snapshot=` | What it is |
| --- | --- |
| *(default)* `public` | Honest project-shaped snapshot: G01/G02-like PASS, models **not released**, full replay NOT_RUN |
| `missing-parent` | Fixture: child names a parent that is not in the docket |
| `superseded` | Fixture: FAIL remains listed after a later PASS |
| `empty` | Fixture: zero checks — shown as an error, not success |

Example: `/#/?snapshot=missing-parent`

## Claims this UI will not make

- Models are available
- Full training replay passed
- `locallyRecomputed` is true (always false here)
- Copyable CLI (withheld until a tested command exists)
- Chat answers come from a released OpenVerifiableLLM model

## Maintainer still needs to provide

- Approved public snapshot + git revision
- Tested verify commands
- GitHub Pages `base` path / URL
- Inference backend after a real release

Do not edit `src/` (pipeline), keys, or `project/` from this app.
