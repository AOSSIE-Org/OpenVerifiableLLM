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
