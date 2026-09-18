# Production numerical execution and public progress gates

These modules implement the shared trajectory and verification drivers. Actual
CUDA production, progress signing/publication, installed-runtime and cost-controller
integration remain unexecuted. Local tests substitute CPU execution and publisher
endorsements explicitly; none completes a production acceptance gate.

`production_trajectory.walk` regenerates initialization after discarded warmup,
checks the registered code/runtime/full initial state, traverses all Wikipedia
updates, resets AdamW at the conversation transition and traverses every conversation
update. Both recording and verification replay use this one numerical iterator.
Primary and recovery checkpoint schedules are kept separate.

`production_record.record` first authenticates source/registration endorsements and
checks the full input census. It saves safe checkpoints, signs each exact boundary
with the registered run key and pauses. Before advancing, `await_anchor` calls the
real Sigstore verifier for the entire current public progress prefix under policies
supplied separately from the downloaded artifacts. Missing anchors stop at the
controller's earlier checkpoint deadline. Invalid signatures or identities fail
immediately. An external controller stop request also preserves the latest state.
No provisioning is performed here; live cost/provider admission belongs to the
operating controller and must be implemented before paid execution.

Explicit recording recovery rechecks signed checkpoint bytes, full stream cursors
and every public progress endorsement before restoring an already committed state.
It regenerates initialization and rechecks runtime compatibility first. Uncommitted
suffix updates are recomputed. A complete orphan checkpoint is adopted only if its
entire state equals that recomputation; partial files are moved to a separately
preserved directory and fsynced before replacement. An altered complete orphan
fails closed. Published checkpoints are never rewritten or discarded.

`production_replay` authenticates registration and the complete public progress
prefix, checks all checkpoint files and full input cursors, then starts from freshly
regenerated initialization. It compares complete state at every primary boundary
without restoring any prover state. It saves its own primary and recovery
checkpoints in a fresh output directory. This version deliberately has no replay
resume reader: these outputs preserve diagnostics/state, but do not themselves
justify skipping a recomputation prefix. Full replay must finish within its budgeted
rental; a failed attempt is preserved and its cost charged before any fresh restart.
Recording recovery is not a substitute for full verification replay.

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python \
  -m ovl_pipeline.production_replay \
  --packet PUBLIC_REGISTRATION_PACKET \
  --registration-bundle REGISTRATION_SIGSTORE_BUNDLE \
  --production-policy EXTERNAL_REGISTRATION_POLICY \
  --source-policy EXTERNAL_SOURCE_POLICY \
  --source-checkout PINNED_CHECKOUT \
  --chain-directory DOWNLOADED_COMPLETE_CHECKPOINT_CHAIN \
  --progress-directory DOWNLOADED_PROGRESS_ANCHORS \
  --progress-policies EXTERNAL_ORDERED_PROGRESS_POLICIES \
  --wikipedia-stream PREPARED_WIKIPEDIA \
  --conversation-stream PREPARED_CONVERSATIONS \
  --output FRESH_VERIFIER_OUTPUT
```

The actual GPU process also requires the complete deterministic environment declared
by `gpu.configure` and the hash-locked CUDA runtime. There is no CPU fallback in the
production path. Independent raw-transformation reconstruction, full public-download
verification, evaluation/inference checks and provider cost accounting remain
separate required checks. A numerical replay report alone does not complete a release.
The model's provenance does not establish generated-answer factual accuracy.

`progress_anchoring` uses an exact, separate `anchor-progress.yml` publisher identity.
Each assertion binds the registration root, exact signed boundary, previous public
assertion and immutable HF checkpoint inventory. The verifier recomputes signatures
and transparency inclusion; saved PASS strings cannot replace them. The associated
signing workflow, remote publication and actual live progress endorsements are still
to be implemented and exercised. The run signature by itself supplies no public gate.

Pilot replay timing now includes saving every compared verifier checkpoint. Parent
checks require that overhead and its complete count, so the full-replay forecast
cannot silently omit it. Initial/transition saves, cold full-input checks, public
anchoring/transfers and reconstruction still belong in measured fixed-cost evidence.
