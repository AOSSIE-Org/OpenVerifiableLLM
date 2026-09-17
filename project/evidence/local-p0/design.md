# P0 design and limits — 2026-09-18

Astra proposal before advisory: introduce a separate versioned pipeline package;
retain legacy experiments and CLI without representing them as full verification.
Use restricted canonical JSON (no floats; binary64 config scalars encoded explicitly),
metadata-aware tensor hashes, named optimizer state, safetensors plus JSON,
immutable signed boundary manifests and an external caller-supplied fixture trust
root. A local synthetic fixture may exercise the complete computational path but
must never be accepted as a publicly anchored production run.

Train and replay share the update kernel. Replay regenerates initialization and
continues through every update without loading prover checkpoints. Independent
coverage checks derive target counts/ranges from document index, including EOS and
partial windows. Boundary digests include optimizer, RNG and control state.

Acceptance for this milestone: deterministic raw-to-prepared regeneration; exact
continuous two-phase replay; resumed equals uninterrupted; source/state/parent/key/
coverage mutations fail; missing production anchoring fails. No G01–G10 passes
are inferred from synthetic fixtures. GPU configuration and throughput remain
unmeasured. Production trust, full corpus and public publication remain pending.

Single writer: Astra owns src/ovl_pipeline, tests and project state. Claude advisory
will inspect a frozen packet read-only; no shared writes, tools or scientific
acceptance. Dependencies are the completed source packet and local test evidence;
review output is kept under this milestone and integrated only by Astra.
