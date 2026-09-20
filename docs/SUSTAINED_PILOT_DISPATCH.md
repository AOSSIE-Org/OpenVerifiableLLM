# Sustained development pilots within a fixed rental

`scripts/run_sustained_pilot.py` dispatches a selected sequence on an **already
guarded single pod**. It never provisions compute, changes the rental deadline,
admits production or declares numerical verification from a process exit.

Each stage selects a complete work interval and a separate complete-export
reserve. Admission resolves the deadline once, writes the derivation before its
job descriptor, and refuses a shorter interval if the fixed rental has insufficient
time. Restart adopts that original deadline and the original launch fence. Replay
parents are selected from fully retained record files and safe checkpoint states;
their complete inventory becomes required input to the replay worker.

The dispatcher retains every declared output before launching a successor. During
a numerical stage it also exports and validates the initial safe state. This
supports the existing durable-export guard; repeated observation of that state
cannot renew the guard. The selected runtime audit, complete stream validator,
numerical kernel, continuous replay and final safe-state checks remain required.
The isolated pilot wrapper adds no shorter numerical timeout or automatic retry.

The separate public-input downloader obtains all fifteen selected preparation,
tokenizer and full Wikipedia/conversation stream files at an immutable public
revision. It reads and hashes every byte. Its transfer counter supplies bounded
liveness only; neither byte movement nor a zero process exit grants verification.
The complete public preparation archive also retains corpus text and transformation
ledgers. Clean raw reconstruction is a separate required operation.

A failed launched stage receives an owned stop request and retains its real
terminal outputs. Invalid telemetry cannot prevent that cleanup. If process
identity or complete retention cannot be established, the dispatcher fails closed
and leaves the unchanged controller and external watchdog authoritative. Automatic
provider termination remains unverified.

Read-only supervision and metadata observations have at most two attempts within
one fixed 45-second window, capped by the original external deadline. SSH retains
error diagnostics in a bounded private buffer: quiet mode would suppress the
information needed to classify a connection refusal. Every diagnostic line must
match the closed transport allowlist before a nonzero SSH result may retry.
Received authentication or host-identity denials fail promptly; unknown or mixed
diagnostics remain fatal, including at timeout. A nonblocking deadline check
includes already-buffered stderr and an available exit status without extending
the window. Raw diagnostics never enter public exceptions or retry receipts.
Diagnostic classification is not proof of its producer or successful execution;
SSH also carries remote stderr. All selected identity, framing, input and state
checks still apply. Launches and writes acquire no automatic retry permission.

The CLI takes the same explicit endpoint, key, worker, controller journal and
health paths as the finite dispatcher, with an `ovl.sustained-pilot-plan.v1` plan:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_sustained_pilot.py --help
```

Development tests cover real local processes and file transfers, original-deadline
adoption, failed-stage retention, altered parents, stale guards, byte counters and
an actual tiny CPU record/full replay with explicit process/CUDA substitutes.
These tests do not establish a sustained CUDA rate, a production forecast or an
independent third-party verification. Measure remote startup, complete hashing,
numerical work and complete exports before freezing production budgets.
