# Persistent checkpoint publication

`scripts/persistent_publication.py` starts one local user systemd service for one
operator-selected checkpoint publication. It does not provision compute, authorize
training, or replace the existing publisher's signature and fresh-download checks.

Build a selection with `selection(registration_root, boundary_root, deadline,
arguments, python=...)` from independently verified, completely retained inputs.
The ten closed arguments are those of `publish_progress_boundary.py`: packet,
registration-bundle, production-policy, source-policy, source-checkout, config,
chain-directory, previous-directory, previous-policies and output. Input directories
must be immutable snapshots; output and service state must be separate from inputs
and each other. Use a pinned source checkout without mutable Git metadata.
The selection inventories every input file, publisher source module and dependency
declaration and hashes the selected interpreter. Inventory construction supplies
integrity, not publisher trust or proof of an installed environment.

Save the canonical selection and independently retain its SHA-256. Start or adopt:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/persistent_publication.py start \
  --selection /absolute/selection.json --sha256 EXPECTED_SHA256 \
  --state /absolute/service-state
```

The service name is derived from that selection digest. The launcher preserves the
unit and a start fence before its only start request. After an uncertain request,
subsequent calls only inspect the original unit. They never repeat `start`, replace
an existing unit, or renew its deadline. Missing evidence, altered inputs, different
unit fragments and unit drop-ins fail closed. An uncertain fence before an actual
start may therefore require explicit reconciliation; it is not safe to start again.

The worker runs outside the coordinator's control group, uses an absolute deadline,
and permits one execution. Systemd adds a shorter fixed runtime bound, no automatic
restart, and complete control-group termination with a bounded SIGKILL fallback.
It does not remain active after successful exit. A parent exit of zero with an
uncooperative child can still produce a failed service result; preserve both facts.
The user manager and machine remain availability assumptions. GPU deadlines and
external watchdog ownership are separate and unchanged.

The worker runs the existing publisher: complete checkpoint upload and anonymous
fresh download, exact append-only Git request, Actions identity verification,
complete previous anchor prefix verification, then public anchor upload and fresh
download. Its result binds the actual acknowledgement file and selected registration
and boundary. A returned receipt or successful service is not permission to advance:
the coordinator must independently select the expected publisher policy, reverify the
complete acknowledgement/prefix, and use the existing ordered checkpoint handoff.
The recorder verifies the public prefix again before its next update.

Development checks use actual tiny CPU safe checkpoints with explicit HF, Actions,
signature and CUDA substitutes. Separate local systemd probes test coordinator exit,
main-process failure, deadlines and uncooperative child cleanup. Neither is live
production acceptance or independent third-party verification.
