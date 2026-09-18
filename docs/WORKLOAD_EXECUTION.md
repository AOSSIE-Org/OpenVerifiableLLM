# Bounded workload execution components

These components have actual local process/file tests and explicit SSH/provider
substitutes. They do not yet establish a live RunPod workload or production
admission. The external cost controller and watchdog retain provider lifecycle
authority; provider automatic termination remains unverified.

`scripts/pod_job_worker.py` is a standard-library runner for one operator-selected
job digest and separately selected worker-source digest. A durable directory and
execution-intent fence precede spawning. Missing receipts permit observation and
export only, never automatic resubmission. The job has explicit argv (no shell),
working directory, an environment allowlist, complete selected input hashes,
absolute deadline, stop grace, free-space floor and export roots. The numerical
executable resolves to a hash-bound regular input. The runner strips inherited
credentials and agent sockets. The audited numerical launcher still owns full
runtime/source/registration checks; starting this wrapper grants none of them.
The setup stage may use the container's interpreter before the public interpreter
is installed; that bootstrap executable is trusted through the selected image
digest. It is not independently hash-bound by the setup exception. Every subsequent
numerical stage requires the hash-bound interpreter and audited runtime.

The runner is detached from its launcher. It records the child PID, start ticks and
process group. It retains the exited leader with Linux `waitid(..., WNOWAIT)`
before terminating remaining members of its owned process group, then reaps it.
See the [Python API contract](https://docs.python.org/3.12/library/os.html#os.waitid).
Deadline, stop requests, log-size limits and insufficient space end the owned
workload. Deliberately escaping into a different session is outside this trusted
workload protocol; this is not hostile-process containment. Provider teardown is
still required even if this runner dies or cannot finish an I/O operation.
Read-only adoption reports the observed supervision state. Recovery acquires the
absent supervisor's lease, checks the retained child identity before signalling,
and records distinct `ABANDONED` status with an unavailable exit code. It cannot
turn an uncertain spawn into a successful exit or silently start the job again.
An empty launch fence is diagnosable and may be explicitly abandoned; unknown
spawn identity still requires external pod teardown and preservation of available
evidence. No retry is authorized by absence of a receipt or directory alone.

`scripts/pod_job_client.py` uploads only the selected worker and job descriptor,
fences the off-pod start request and reconciles uncertain responses read-only.
It exports every regular file in each caller-selected output root, including empty
logs, and compares a second complete remote inventory before issuing a transfer
receipt. Symlinks, changed bytes, new files and incomplete copies fail closed and
remain preserved. This establishes selected-byte preservation, not numerical
correctness. Private signing material must remain outside these export roots.

`scripts/workload_health.py` runs under the persistent coordinator's exclusive
journal lease. Its heartbeat uses the controller host's clock. Repeated runtime
sequences, unchanged control, retried byte counts below their retained high-water
mark and repeated identical exports cannot reset progress/export age.
Bulk transfer progress requires at least 1 MiB of new bytes or completion of the
selected file, with immutable size/direction and restart-preserved credit. Status
polls do not count as bulk transfers. New runtime
sequences may reflect paid warmup work followed by regenerated initialization;
they are liveness observations, not monotonic coverage proof. Remote timestamps
are never used. Same-host restart preserves the original lifetime; clock rollback
or reboot refuses renewal.

Export health requires actual local inventory verification. Terminal job records
must be preserved, every prior selected export is rehashed before final completion,
and an adopted completion rechecks those files. `complete: true` authorizes prompt
cost teardown even when a job failed and its failure artifacts were saved. It never
means model or replay acceptance.
Completion evidence is rehashed by `finish()` and once on adoption by a new health
writer. Subsequent completion heartbeats in that process reuse that check; exported
directories must remain immutable. The generic health API currently rejects
production recording and full production replay until their state-aware retention
checks are integrated, so a log-only generic export cannot finalize those jobs.

`scripts/run_workload_stage.py` joins these components for one short setup, pilot
or export stage within an already adopted rental. It observes real input-hash
prefix progress and optional completed-update telemetry, writes health, delivers
the controller's stop marker, waits for terminal status, exports all selected
roots and preserves a result before marking the stage finished. It includes the
job-record directory automatically; embedding the job digest inside its own
input manifest would create a digest cycle. A saved stage is rehashed and adopted
without another remote start. A prior stop forbids launching a new stage.

Mutable status uses one bounded open-file read. The former inspect-then-fetch
approach could race atomic replacement; a real stop fixture exposed that failure.
Immutable checkpoints still use independently selected inventories and full hashes.

The stage API deliberately rejects production recording and long full replay
until their periodic verified checkpoint hooks are integrated. It caps a new
unhooked stage at 1500 seconds, leaving time before the cost controller's
1800-second export-age limit. The enclosing persistent coordinator must select the
next stage, enforce all actual admission gates, preserve final exports and finish
health only after every planned stage exits. Live SSH/runtime admission, long-run
checkpoint dispatch and actual CUDA measurements remain outstanding.

## Persistent finite coordinator

`run_workload_coordinator.py` runs under its own exclusive health journal lease.
Its independently selected v3 plan binds the exact rental intent, SSH profile,
worker source, immutable setup uploads and ordered finite job descriptors. It
reads an atomic hash-linked prefix of the existing rental controller journal,
requires the one attributed pod and a fresh armed watchdog before new stages,
and never calls provider creation. Stopping rentals cannot start another job.
Each actual input transfer advances health only by the bounded byte rule and has
a retained receipt; the selected worker rehashes required inputs before use.

On restart the coordinator adopts job/export receipts. It cannot repeat a fenced
launch, change the selected plan or discard a failed stage. A nonzero/abandoned
stage stops the rest of the finite plan after complete selected exports. Final
reports retain exact stage-result digests and terminal records. Interruption
between the final report and health completion reconstructs the report's inputs
before completing; it does not rewrite the report to match a later stop request.
Completed adoption rehashes retained artifacts and does not issue a new live
heartbeat after the rental deadline.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_workload_coordinator.py \
  --plan PLAN.json --plan-sha256 INDEPENDENTLY_SELECTED_PLAN_SHA256 \
  --rental-intent RENTAL.json --controller-journal RENTAL_JOURNAL \
  --watchdog-heartbeat WATCHDOG_JOURNAL/heartbeat.json \
  --profile PROFILE.json --key PRIVATE_SSH_KEY --known-hosts PINNED_KNOWN_HOSTS \
  --inputs LOCAL_SELECTED_INPUTS --worker scripts/pod_job_worker.py \
  --output COORDINATOR_DIRECTORY --health WORKLOAD_HEALTH.json
```

The coordinator belongs in a user systemd unit with `Restart=on-failure`, a finite
restart rate and the unchanged pinned arguments. Both external provider guards
must already be armed. Setup upload time and all failures remain paid rental time;
source hashing or installer work cannot extend the original deadline. Actual
service/live-provider validation and production hooks remain separate checks.

`pod_runtime_setup.py` prepares the target runtime entirely offline. The selected
container interpreter starts it with `-I -S`; its only parent dependencies are two
individually pinned public pure-Python wheels. It verifies the complete selected
source tree and every locked wheel, extracts the unmodified public Python archive,
creates an empty target venv, installs exact local wheel paths with hashes and
network indexes disabled, then audits every installed payload and tests constrained
startup. Replacing a direct-URL lock row with its already verified local wheel path
prevents pip from downloading that URL despite `--no-index`. This derives the
installer input from the complete checked wheel set, without changing the numerical
lock. Setup partials require preservation and a fresh attempt directory.

The prospective finite plan format is v3. It requires explicit export reserve,
transfer/hash planning floors and a stated timing basis; an estimate is not a live
measurement. Static validation includes the full worker descriptor, isolated
image-Python bootstrap allowlist, at most one initial setup stage, declared output
roots disjoint from inputs/tools, and exact upload-to-required-input equality before
every stage. Finite jobs must leave their export reserve before graceful shutdown
and fit within the initial 1800-second export window. This deliberately limits the
whole finite plan; longer production jobs need dedicated periodic hooks. Export
budgeting includes six full local hash passes rather than treating inode/mtime as
byte verification. Live output sizes and transfer/hash rates still require measured
admission before production.

Each upload is bounded to 180 seconds at its declared planning floor and receives
at most that duration plus 30 seconds per attempt, inside the original graceful
stop time. Retries retain byte high-water marks; restarting a transfer cannot
manufacture progress. Slow links fail the bounded attempt instead of renewing the
rental indefinitely. Failed or delayed setup still consumes the paid allocation.

The bootstrap refuses all source bytecode/cache paths before parent imports, and
checks each helper wheel against its own exact pure-Python dependency-lock row
using only stdlib. This prevents an unchecked-hash source cache from bypassing the
selected source inventory. The generated pip file binds the already audited wheel
set; the pre-install wheel-manifest check and post-install payload check establish
its numerical-lock relationship. The coordinator is a trusted process on the
same owner host as the provider guards; it is not a credential-isolated capability
sandbox. Importing its provider helper does not itself read credentials or call
the provider. Its code does not create resources.

Numerical activity is reported by the selected trusted workload and observed over
SSH; it is not independent of that workload. Absolute lifetime/budget guards remain
independent of those reports. Log growth would not prove useful numerical work and
is not accepted as corroboration. Full replay remains the computation check.

## Production retention components under integration

`pod_versioned_export.py` preserves a complete stable peer tree in each snapshot,
using read-only hardlinks for identical verified bytes. It still inventories and
hashes the full selected tree before and after transfer, and verifies local bytes.
Different versions and failed partials remain preserved. Hardlinks are shared
physical storage, not independent backup copies or hardware attestation.

`production_retention.py` requires a terminal worker observation, then exports all
descriptor-declared roots plus the job records through separately pinned profiles
for the same endpoint. It includes primary states, recoveries, partial files and
metadata, and compares terminal records before/after the exports. Its verifier
rejects omitted roots, mismatched profiles and changed local bytes. Logs alone
cannot establish terminal retention.

`production_health.py` adds retention-bound completion and finite publication
advancement to the generic health journal. Publication activity must select an
actual signature/state-checked checkpoint snapshot; repeat stages cannot renew
health and the original publication deadline is retained. The publisher emits
only completed gates and identified successful Actions step transitions. These
are liveness observations, not signature or training verification. Production
health rehashes every declared retained output before final cost completion.
The enclosing production dispatcher is not yet integrated; these components do
not by themselves enable a production rental or mark any acceptance gate passed.

The finite output size bound is a planning admission constraint, not a remote filesystem quota. Actual oversized exports are retained and prevent any next stage or completion. The independent fixed rental deadline remains authoritative if export cannot finish. Every stage deadline is conservatively at most 1500 seconds after the original rental start. Controller journal prefix digests observed before uploads and stages are retained.
