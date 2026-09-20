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

Launch acknowledgement uses one bounded metadata exchange for the exact intent,
runner receipt and child receipt, followed by process supervision. Each file keeps
its own framing, hash, canonical JSON and identity checks. This is not an atomic
snapshot and grants no numerical verification. The local launch fence still
forbids sending a second start after an uncertain result.

Future dispatchers select a 120-second transfer deadline for launch/adoption,
clipped to both the existing job and graceful-stop deadlines for a new launch, or
the existing external termination deadline for read-only adoption. Local hashing,
persistence and subprocess cleanup are not a strict whole-operation elapsed-time
bound. Eligibility is checked again after uploads, immediately before the launch
fence and again before dispatch. This is not atomic with an external stop writer
or remote command arrival: the worker's original job deadline and the external
rental teardown remain authoritative. Pending stops on fenced adoption are sent
before potentially lengthy receipt reads. Input hashing checks stop/deadline conditions between chunks and before
workload spawn. The selected job,
phase and rental deadlines do not move. The cost supervisor's 300-second stall
limit and strict authentication/identity/integrity failures remain unchanged.
The allowance accommodates measured multi-exchange startup latency; it does not
award progress for waiting or authorize mutation retries.

A received start acknowledgement must match the selected job, worker and process
identity and agree with the retained remote receipt. A missing acknowledgement
may be reconciled read-only; a retained contradiction fails again on adoption.
Received raw reply bytes are retained even if the transport subsequently fails;
nonempty malformed replies remain unresolved. Completion and abandonment signal
only an identity-bound process group and check for surviving group members before
writing a terminal receipt. An absent leader without established group ownership remains
unresolved and requires external teardown. Deliberate process-group escape is
outside the trusted workload contract.

The bounded read-recovery policy adds no retry to writes. Existing immutable
input and stop-file delivery can be attempted again on coordinator re-entry after
a lost acknowledgement: the receiver hashes the complete existing bytes and
rejects different content rather than overwriting it. This idempotent file-delivery
behavior does not permit another workload start or provider creation.

Worker stop requests use one canonical payload bound to the selected job digest.
Controller and dispatcher causes remain separate local records; they do not
compete to overwrite the worker's immutable stop file. Exact existing bytes are
accepted by the original immutable transfer check. Foreign or changed bytes,
including incompatible older payloads, remain errors. This is a prospective
protocol change: it does not authorize changing a frozen attempt's source.
A numerical recording stop remains separate and precedes its hard worker stop.
A graceful stop followed by the controller's immutable stop retains the original
numerical marker and first timing intent, records the later cause separately,
and can only shorten the hard-stop deadline. Saved delivery receipts bind the
job, complete endpoint profile, destination and exact payload; damaged receipts
cannot suppress delivery. These operations require the existing single coordinator
lease. Sequential re-entry is supported; simultaneous marker installation is not.
The coordinator rechecks time after numerical delivery and clips adoption reads
to a pending hard stop. Remote arrival is not guaranteed at that scheduling bound;
the frozen worker deadline and independent rental termination remain authoritative.
Failure retention also preserves an existing terminal-export deadline.

Failed read observations and enclosing dispatch failures can retain diagnostics under the caller's private
`private-transport-diagnostics` directory. These files are outside the declared
remote exports and must never be included in public artifacts. They record the
operation phase, selected identities, exception locations, bounded stderr and
received metadata prefixes, and any observed cleanup error. They do not dump
commands, environments or transfer inputs. Each byte prefix is capped at 64 KiB;
each record at 256 KiB; each stage directory at 128 records and 32 MiB. Directories
and files require owner-only permissions. Truncation is explicit. Diagnostics are
best effort: storage failure preserves the primary exception and changes neither
retry classification, original deadlines nor verification credit. An unresolved
transport cleanup failure prevents another read attempt. Missing
historical diagnostics do not justify an inferred root cause.
