# Proposed PR

**Title:** Add a static evidence explorer and release frontend for OpenVerifiableLLM

## Description

Visitors need to distinguish model provenance, scoped replay reports and model availability. This adds an isolated React/TypeScript/Vite frontend featuring an enhanced design system with an interactive 3D globe, container scroll perspective cards, six responsive views, URL-preserved evidence filters, shareable records, parent/supersession navigation and full-digest copying.

The application uses validated, explicitly synthetic presentation metadata until maintainers supply an approved pinned public snapshot. Invalid/empty/missing metadata cannot become overall success. Verification results remain separate from workflow progress and browser loading. Both current model roles stay unreleased and production generation stays unavailable.

Includes a development-only inference adapter/UI with exact release/settings checks, accessible interaction states, a lockfile, unit/component/browser tests, production mock-exclusion checks, static base-path routing and an integration handoff. No training, verifier, trust policy, signing, progress state, Python dependency or cloud-resource change is included.

## Validation

See VALIDATION_REPORT.md for actual executed checks and screenshots/. Do not claim the upstream Python suite ran; it was not run for this isolated frontend deliverable. The current full upstream checks should be run in the integrating contributor's supported checkout before PR submission, as CONTRIBUTING.md requests.

## Review focus

- Presentation contract and claim/attribution wording.
- Evidence fixtures and preservation of failed/superseded originals.
- Approved public metadata mapping before changing data mode.
- Frontend-only dependency policy and accessibility checks.

## Remaining maintainer inputs

1. Approved small display snapshot, immutable source revision and raw field mappings.
2. Current PROJECT_GOAL.md and goal_state.json context; no mutable operational state should be published wholesale.
3. Final base/chat repositories, immutable revisions, release roots, inventories, sizes, licenses, model cards, parent relation and supported runtime.
4. Approved tested instruction text for each verifier profile, prerequisites, validation evidence and sourced resource measurements/estimates.
5. Supported inference interface, release identity/capabilities, limits, privacy policy and cancellation semantics.
6. Static-host owner/target/base path and any approved branding assets.

## Changed paths

All delivered paths are under frontend/: src/, public/data/, tests/, scripts/, screenshots/, licenses/, package.json/package-lock.json, TypeScript/Vite/Vitest/Playwright configuration and documentation. dist/ is included in the export for preview and is ignored by Git; do not commit it unless the upstream host explicitly requires generated files.

The supplied archive had no frontend. Compare the current upstream tree before integration. Work on a separate branch and preserve others' work. No PR has been opened or submitted by this task.

## AI assistance

Implementation, fixture/test authoring, documentation and Figma design translation were AI-assisted with Codex and the Figma plugin. No human review, maintainer approval or real model verification is claimed. Recheck current contribution policy before submitting.
