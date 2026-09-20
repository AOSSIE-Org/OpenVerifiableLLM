# Production checkpoint retention and handoff recovery

`production_checkpoint_poll.CheckpointRetention` operates inside a caller-owned
cost-health journal lease for an already selected production recording or full-replay
job. It does not provision, launch or restart numerical work. The caller must first
authenticate the registration and select the exact job, transport and full recorded
chain required for replay.

Each poll reads peer metadata, validates the complete signed recording prefix or the
replay session/comparison prefix, and selects its latest complete primary or recovery
checkpoint. The existing live-retention helper copies and validates all safe-state
files. It verifies checkpoint control and state identity before reporting export
health. A checkpoint receipt is neither numerical replay nor job completion.

A durable observation journal rejects regressing checkpoints and changed checkpoints
at the same training position. The valid Wikipedia-base to conversation transition
has the same global update count with a different phase and remains supported.
Restart reconstructs this history and rechecks retained bytes. Repeated polls cannot
renew progress/export credit. Every checkpoint copy keeps its first selected deadline;
one preserved incomplete copy may receive one fresh copy attempt within that same
bound. Malformed completed receipts never trigger replacement.

The poller may observe a later checkpoint after a fast producer advances. Complete
terminal retention still copies **every** declared output, including earlier primary,
recovery, audit and partial files. The latest live copy cannot replace that check.
No full-replay resume reader is provided; retained verifier states do not imply that
loading them is supported or that a partial replay passes the complete trajectory.

`reconcile_checkpoint_delivery.reconcile` handles uncertainty after the coordinator
may have delivered a public progress policy but lost its receipt. It takes the same
independently selected policies as the normal handoff. It rechecks the local complete
checkpoint and public signature prefix, then reads the peer's signed chain and policy
file. An exact policy prefix requires downloading and verifying every peer anchor
file, followed by another policy read to reject concurrent replacement. It never
writes remote files, rolls back policy, or fabricates a recorder-consumption claim.

A recorder may already have removed its old waiting marker; recovery therefore does
not require that obsolete marker. Its signed chain must preserve the selected prefix
and extend by no more than the next boundary. A prior policy still present yields
`NOT_DELIVERED`, allowing only a separately guarded current-boundary handoff. An
unrelated, changed, or unexpectedly advanced policy fails closed.

Local tests use actual tiny CPU safe states, signatures and subprocess file transfers
with explicit publisher/CUDA substitutes. They cover complete primary/recovery state
retention, actual fresh CPU replay outputs, interrupted copies, lost delivery receipts,
phase transitions, source/parent/state changes, partial policy delivery and read races.
These helpers are not an enclosing production coordinator or production admission.
