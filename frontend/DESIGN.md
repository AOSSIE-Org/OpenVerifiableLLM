# OpenVerifiableLLM public site — UI plan

For: ryoari / AOSSIE OpenLLM. Owner of `frontend/` only.
Stack: React, TypeScript, Vite, ordinary CSS. Static. No backend.
Models: **not released**. Chat is a labelled shell, not a product.

This is the product cut of `FRONTEND_CONTRIBUTOR_BRIEF.md`. The brief is the
list of things we must not lie about. This file is what a visitor actually sees.

---

## One-line design read

A public **evidence docket** for a research release that does not exist yet:
calm, typeset, forensic. Not a chatbot landing, not a SaaS dashboard.

If someone closes the tab they should remember the **claim strip** — a persistent
sentence of what this page is allowed to say — and a **chain with a missing
link** when a parent is absent.

---

## What the site is for

A visitor should leave able to answer:

1. What is OpenVerifiableLLM trying to prove?
2. Which public sources and transforms are in play?
3. What was actually checked, by whom, in what scope?
4. Where are the original artifacts?
5. Are the models out? (today: no)
6. Later: how do I download, verify, and run the *exact* weights?

If the homepage looks like ChatGPT, we failed the assignment and the org.

---

## Visual lock

Do not use Inter, purple glow, glass cards, three feature tiles, or a giant
green “VERIFIED” badge.

| Token | Hex | Use |
| --- | --- | --- |
| Paper | `#F3F0E8` | Page ground (warm, not cream-terracotta UI kit) |
| Ink | `#1B2320` | Body and headings |
| Mute | `#5E6A64` | Secondary text |
| Rule | `#D4D0C4` | Hairlines, table rules |
| Pass | `#2C6A4E` | Scoped pass only — never a page background |
| Fail | `#9B3B32` | Scoped fail |
| Hold | `#A67C2D` | NOT_RUN / pending / not released |
| Link | `#215A8F` | Evidence and external refs |

**Type**

- Display / titles: `Source Serif 4`
- UI / body: `Source Sans 3`
- Digests, commands, IDs: `Source Code Pro`

**Signature element (one place, then quiet)**

A **claim strip** under the header on every route:

```
This page may claim: data reconstruction (project-operated) PASS
It may not claim: a released model · full training replay · answer accuracy
Snapshot 2026-09-19 · source commit abc1234
```

Evidence pages add a **chain rail** on the left of the detail view: parent →
this object → children. A missing parent is an empty notch with “link absent”,
not a skipped row.

Motion: 150–200ms opacity/transform on expand and copy. Honor
`prefers-reduced-motion`. No looped animation.

---

## Reference kits (`I am giving you these repositories.txt`)

Those six libraries are **interaction textbooks**, not a look to clone:

Motion Primitives · Magic UI · Aceternity UI · React Bits · Cult UI · Animate UI

The same note also says avoid AI slop, random animation, and generic shadcn.
Taken literally, “cinematic + 3D + cursor-reactive + glass + futuristic AI
product” **is** that slop, and it fights ryoari’s brief (restrained research
site, no decorative dashboards).

We steal craft. We do not steal chrome.

| Steal | Where it lands | Refuse |
| --- | --- | --- |
| Staggered reveal of the pipeline steps | Overview rail, 40ms stagger, once | Full-page cinematic intro |
| Hover/press on copy, filter, row | Digests, table rows, buttons | Custom cursor, magnetic buttons |
| Height/opacity on expand | Evidence JSON, claim details | Accordion everywhere |
| Route fade (opacity + 8px) | Between the five pages | Shared-element 3D page flips |
| Skeleton shimmer *only* on evidence fetch | `/evidence` loading | Spinners on static copy |
| Spatial hierarchy: one column, deep type | Whole site | Bento grids, glass panels |
| Animated *state* (PASS appearing, FAIL flash) | Result marks | Looping background shaders |
| Scroll: pipeline pins then releases | Overview only, optional | Scroll hijack, parallax dust |

Implementation rule: `transform` and `opacity` only, 140–220ms, `ease-out`.
`prefers-reduced-motion` drops travel. No `transition: all`. No particles.
Do not add Aceternity/Magic UI as dependencies; reimplement the two or three
patterns we need in CSS.

**The ASCII diagram in that file is a different product.** It is:

```
user query → LLM → evidence retrieval + claims extraction
        → verification → provenance graph → verified response
```

That is a **RAG / claim-checking chatbot**. OpenVerifiableLLM is the opposite
direction: prove the *model and training*, not each chat sentence against a
Wikipedia article. The contributor brief forbids RAG and article-attribution.
Do not build that graph as the homepage. The provenance we show is the
**training evidence chain** (dump → extract → train → replay → release).

If `/try` ever exists for real, a response may link to *model identity*
(release root, decoding recipe), never “this token came from page 4412.”

---

## Information architecture

Five routes. Nothing else in v1.

| Route | Purpose |
| --- | --- |
| `/` | What this is, current honest status, three next actions |
| `/evidence` | Filterable docket of public checks and artifacts |
| `/evidence/:id` | One object: digest, parents, children, original link |
| `/releases` | Base card + chat card, both gated until real metadata |
| `/verify` | Five verification *profiles*, not a magic button |
| `/try` | Inference shell. Public: disabled. Fixture preview: labelled mock |

Nav: Overview · Evidence · Releases · Verify · Try  
Footer: AOSSIE, GitHub, Hugging Face evidence dataset, snapshot time, “this is
a viewer, not a verifier.”

---

## What must be on each screen

### 1. Overview `/`

**Needed**

- One paragraph in plain language: a small model whose *declared* public
  inputs, transforms, and training steps can be inspected and replayed.
- Availability line in Hold color: **No model release yet.**
- Three actions only: Explore evidence · View releases · How verification works
- Pipeline as a **vertical sequence** (this *is* a sequence, so order is allowed):
  Wikipedia dump → extract/exclude ledger → tokenizer/streams → init →
  Wikipedia training → conversation training → full replay → publication
- Each step shows one of: PASS / FAIL / NOT_RUN / UNAVAILABLE / UNSUPPORTED
  plus **scope** (fixture | pilot | production) and **who** (publisher report vs
  project-operated vs not done)
- One box: **Provenance is not truth.** Training on Wikipedia does not make a
  sentence a citation.

**Do not**

- Progress percentage (“G01–G02 = 20% done”)
- Fake completion date
- Fake assistant reply in the hero
- A site-wide green badge

**Craft**

The sequence is the page. No card grid. Type + a left rail of status ticks.
The claim strip is the hero, not a slogan.

### 2. Evidence explorer `/evidence`

This is the product. Mentors will judge GSoC potential here.

**Needed**

- Table/list as the primary UI (brief requires this; diagram is optional extra)
- Filters: phase, kind, result, scope (fixture / pilot / production)
- Columns: title · phase · scope · result · performed by · attested by · when · source
- Click row → detail
- Empty, loading, fetch-fail, and “zero checks” states that **cannot** look like success

**Groups** (filters, not eight separate apps)

1. Wikipedia source + acquisition  
2. OASST / conversation source  
3. Extraction + inclusion/exclusion  
4. Tokenizer + streams  
5. Init + frozen config  
6. Training boundaries / checkpoints  
7. Reconstruction + replay reports  
8. Release artifacts + clean download  

**Craft**

- Result is a word plus a mark, never color alone
- Fixture rows have a persistent “FIXTURE” tag so they cannot be skimmed as production
- Superseded reports stay listed, struck through, with a link to the successor
- Do not fetch dumps, shards, or checkpoints. Show **size + download link**

### 3. Evidence detail `/evidence/:id`

**Needed**

- Title, kind, scope, result, explanation
- Full digest with copy (truncate visually, copy full)
- `performedBy` / `attestedBy` / `locallyRecomputed` as three separate fields  
  (`locallyRecomputed` is always false on this static site)
- Parent IDs and child IDs; missing parent = broken chain, not omitted
- Link to original (GitHub blob, HF file, commit)
- Expandable escaped JSON, bounded (e.g. 64 KB). Not a second verifier.

**Craft**

Chain rail is here. If parent is missing, the rail has a gap you can see without
reading the table. That screenshot is the one that belongs in the PR.

### 4. Releases `/releases`

Two panels, stacked on mobile, side by side on desktop: **Base** and **Chat**.

**Until maintainer metadata exists (today)**

- Availability: **Not released yet**
- Download and Generate controls present but **disabled**, with a sentence next
  to them: why, and what will unlock them
- Chat panel states it is a derivative of base, so it cannot pretend to be
  “Wikipedia-only”

**When a real release exists**

- Role, repo, immutable revision, release root
- Param count, file sizes, loader — only from metadata
- License/attribution from that release
- Chat → exact base parent
- Verified *scope* (which profiles passed), links into evidence
- Download instructions + limitations

**Do not** invent HF repo names, leaderboard scores, or placeholder `.safetensors`.

### 5. Verify `/verify`

Five profiles as five blocks. Each block:

- Name  
- What a PASS is allowed to mean (one sentence from the brief table)  
- What it does **not** mean  
- Status of *this project* for that profile, from the snapshot  
- Command area: **Instructions pending release** until ryoari gives a tested
  command. No copy button on a guessed CLI.

**Needed warning:** complete replay is not a free browser check. If we ever show
cost/time, label estimate vs actual bill.

**Craft**

This page teaches claim hygiene. It should feel like a legend for the rest of
the site, not a marketing FAQ.

### 6. Try `/try`

Public build:

- Model picker visible, both options disabled
- Composer visible, send disabled
- Banner: **Generation is unavailable until a verified model release is connected**
- No silent fallback to GPT / Gemini / “any Ollama model”

Fixture / preview build (`?fixture=1` or a documented env flag):

- Mock adapter only
- Every response carries a visible, sticky label:  
  **Example UI response — not generated by a released OpenVerifiableLLM model**
- Base = completion box; Chat = role-aware thread
- States: empty, loading, error, retry, identity mismatch (adapter returns a
  different release than selected → reject)
- Copy response. Receipt download only when a real backend exists
- No prompt logging, no analytics, no localStorage of visitor chats in v1

This is how we honor “he asked for a chat interface” without shipping a lie.

---

## Status vocabulary (render exactly)

Verification result ≠ workflow ≠ browser state.

| Result | Meaning on screen |
| --- | --- |
| PASS | This named check passed, inside its reported scope |
| FAIL | It ran and did not meet requirements |
| NOT_RUN | Not performed |
| UNAVAILABLE | Evidence or resources missing |
| UNSUPPORTED | Cannot be done on this environment/format |

Never map arbitrary strings like `ok` / `complete` / `true` to PASS.

Always show, when a check is displayed:

- profile · artifact · scope · who performed · who attested · locally recomputed?

Independence is not inferred from “two machines” or “a signature.”

---

## Data: viewer, not verifier

- Versioned presentation snapshot in `frontend/public/data/`
- Adapter maps maintainer JSON → view types. Keep source URLs/revisions.
- Two modes: `fixture` and `public-snapshot`. Failed fetch **must not** fall
  back to a successful mock.
- `goal_state.json` is navigation for *him*, not a public API. Do not scrape it
  from the client. He supplies an approved snapshot when he has one.

Minimum view types (implement these, not a fantasy backend):

```ts
type CheckResult = 'PASS' | 'FAIL' | 'NOT_RUN' | 'UNAVAILABLE' | 'UNSUPPORTED';
type EvidenceScope = 'fixture' | 'pilot' | 'production';
type ReleaseAvailability = 'not-released' | 'available' | 'withdrawn';
```

v1 fixtures (enough, not twelve):

1. Public-shaped snapshot: G01/G02-like PASS, models not-released, replay NOT_RUN  
2. Fixture-only success that is visually unskippable as fixture  
3. FAIL hash + missing parent  
4. Superseded FAIL then later PASS  
5. Empty check list (must not look green)  
6. Fetch error / stale snapshot  

---

## What we will not build (even though we have “freedom”)

These would look busy and would violate the brief or AOSSIE’s less-cloud stance.

- Accounts, wallets, comments, stars
- Training / RunPod / cost control panels
- Browser-side full replay or auto-download of weights
- RAG, “this sentence came from article X”
- Live Ollama against unverified local models as the default path
- Invented `ovllm` CLI on a copy button
- Polling dashboards, graphs of GPU utilization
- A second design system or a UI kit theme

Freedom is in **how clearly the docket reads**, not in extra product surface.

---

## Execution bar (this is the GSoC-looking part)

Mentors remember restraint plus finish, not a 20-page mock.

- Keyboard: skip link, focus ring, no trap in dialogs
- Status not color-only
- 200% zoom, mobile table that becomes stacked definition lists
- Dates with timezone; sizes with units
- Copy digest announces “copied” to the button, not only visually
- External links marked external; hashes never re-serialized as “the same bytes”
- `npm run dev|typecheck|test|build` documented and green
- GitHub Pages base path considered (`basename` / Vite `base`)
- Screenshots in the PR: desktop + mobile, including **not released** and
  **missing parent**

---

## Build order

| Cut | Ship | Why |
| --- | --- | --- |
| M1 | Routes, tokens, claim strip, two fixtures, overview + evidence list | Proves we understood the brief |
| M2 | Detail chain rail, releases gated, verify legend | The memorable UI |
| M3 | One maintainer-approved public snapshot adapter | Honesty with real links |
| M4 | `/try` disabled + labelled mock | Honors the Discord ask without a fake model |
| M5 | a11y pass, tests, static build, README handoff | Mergeable |

Do not wait for Wikipedia training to finish. Do not wait to implement all
fixtures before opening a draft PR.

---

## Questions for ryoari (only these)

1. Confirm v1 is the **evidence site**, with `/try` gated, not a chat app.  
2. When can we have one **approved public snapshot** (commit + files), vs fixtures only?  
3. Hosting: GitHub Pages under AOSSIE, or wait?  
4. Do not send CLI copy until he pastes a command he has actually run.

---

## Why this is the right amount of work

The org does not need another Ollama skin. It needs a public page that can sit
next to a GSoC demo and not overclaim. That is rare, it matches G09 in
`PROJECT_GOAL.md`, and it is entirely inside `frontend/`.

Done looks like: a stranger opens `/`, reads that models are unreleased, opens
evidence, copies a digest, follows a broken parent, and understands the
difference between “we hashed the dump” and “we replayed every step.”
