# Revised RunPod lifetime guard

The revised operating policy requests `terminateAfter`, maintains
an external watchdog and explicitly terminate a still-running pod at deadline +
120 seconds; verify absence and reconcile actual charges including grace.
Automatic provider termination remains **UNVERIFIED**. New selections use a $130
aggregate cap, $120 operating threshold, $10 protected reserve and one-GPU limit.
Historical selections retain their original pinned source and deadlines.
Operator authorization records are private; public reports describe the technical
policy and observed behavior without publishing personal conversations.

`supervision.rental_plan` v2 separately budgets the 120-second grace and at least
300 seconds of billing slack, then moves the requested deadline earlier when funds
require it. Graceful checkpointing begins at least 300 seconds before that deadline.
Past v1 plans and the failed automatic-termination probe remain historical evidence.
A quote must include compute and storage; elapsed setup/failed attempts count.

The standalone watchdog never creates a pod. Its caller pins a canonical intent,
including a v2 budget plan, unique attempt name, one GPU, immutable image, disk sizes,
exact requested deadline and fresh empty-account baseline. Start it off-pod before
creation, under a persistent systemd service with restart-on-failure and append-only
logs. The creation controller must verify a fresh ARMED heartbeat bound to that exact
intent/plan and journal a single creation request. `run_rental_controller.py`
implements that integration; actual live deployment and workload/export dispatcher
integration remain pending. This document and unit tests do not admit production.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_external_watchdog.py \
  --intent CANONICAL_INTENT.json --intent-sha256 EXTERNALLY_SELECTED_SHA256 \
  --journal OWNER_CONTROLLED_FRESH_DIRECTORY
```

Only the unique attributed resource may be terminated. The watchdog adopts a prior
journal and never reissues creation. It uses a monotonic lifetime limit, avoids
starting a potentially slow read across a known-ID deadline, and sends termination
before another read when already due. Observation/storage/shape failures trigger
an earlier abort. A broken journal permits a minimal authenticated exact-name
teardown attempt while preserving the broken evidence; a duplicate lease contender
leaves the live controller alone. Termination requests are retried, never treated as
proof of absence. Two fresh absent observations at least 15 seconds apart are
required, with residual storage and pending billing reported separately.

The external host and provider API remain dependencies. This software cannot
promise termination if the entire external host loses power or the provider API
becomes indefinitely unavailable. No automatic-provider guarantee or independent
third-party verification is claimed. The ordinary cost/progress supervisor must
also enforce spend, forecast, watchdog liveness, checkpoint/export health and early
shutdown; the lifetime watchdog alone is not complete execution admission.

Development checks include fake-provider deadline/grace, transient failures,
misprovisioning, clock rollback, crash adoption, corrupt journals and duplicate
leases. A real systemd process was killed and restarted with the same journal; its
provider and advanced deadline were deliberately simulated, with zero paid actions.

## One-shot rental and spend controller

Start the watchdog first. Run the separate controller under its own restartable
off-pod service with the exact pinned creation intent and a distinct journal:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_rental_controller.py \
  --intent RENTAL_INTENT.json --intent-sha256 SELECTED_RENTAL_INTENT_SHA256 \
  --journal RENTAL_JOURNAL --watchdog-heartbeat WATCHDOG_JOURNAL/heartbeat.json \
  --workload-health DISPATCHER_HEALTH.json
```

The rental intent embeds the watchdog intent and closed, single-GPU, secure-cloud
SSH payload, plus retained quote bytes. `rental_quote.py` checks the selected GPU
and rate against the complete captured provider catalog response, binds its digest
to the plan, and checks the exact private authorization record. It derives the
all-in rate from compute and both disk allocations, with a 25% margin. Storage uses
the [published RunPod rates](https://docs.runpod.io/pods/pricing), the higher stopped
volume rate and a conservative 28-day month. This is an operator-captured quote,
not a signed offer; capacity, cheapest measured configuration and actual charges
still require their separate checks. No credentials or arbitrary environment fields are accepted. A fresh
empty-account observation and armed watchdog are required before durable creation
intent publication. That journal event authorizes exactly one request: even a
crash before sending it requires reconciliation, never automatic recreation. Names
must end in a UUID, and a private account-wide lease plus a persistent per-attempt
creation fence prevents a second journal directory from authorizing another request.

On restart, the controller adopts the attributed ID or unique intent name. It
reserves the larger of account debit and prior unsettled charges plus upper-rate
lifetime accrual, retaining mandatory future-work reservations. Settling an already
reserved predecessor charge does not count against only the new rental's ceiling. Available
balance rounds down and costs round up. Account, shape, watchdog and evidence
failures cause early termination. A transient 429/5xx/transport read receives only
the existing bounded read grace, never beyond the absolute deadline. Auth, identity
and shape failures receive no such grace. Ordinary progress/checkpoint limits and
invalid workload-health files request a
graceful stop, whose fixed deadline cannot be renewed by later heartbeats.
Absent workload health cannot keep idle compute indefinitely. Completion requires
the dispatcher's final export observation before prompt teardown.

The trusted external dispatcher must write health only after actual progress and
durable export checks, translate its local `stop-request.json` into the worker's
`jobs/<selected-job-digest>/request-stop` marker (create once), deliver a separate
root `request-stop` to an active production recorder through its handoff profile,
and enforce the
production registration gates before launching updates. The rental controller
does not launch training, authenticate those gates or turn a health assertion into
artifact verification. Missing dispatcher integration remains an execution gate.
Final termination requires two fresh absence observations at least 15 seconds
apart; actual provider billing reconciliation remains a separate required step.

Both guards persist a deadline in `CLOCK_BOOTTIME` with the host boot identity.
A restart cannot renew that deadline, and a new boot or missing legacy clock
anchor causes teardown. Known resources are terminated before a potentially slow
read when due. Exact UUID-name duplicates trigger cleanup of all attributed copies;
unrelated names remain untouched. A corrupt rental journal receives the same
best-effort exact-name cleanup as the watchdog while preserving damaged evidence.

Health files use `canonical.write_json` (RFC 8785 plus atomic replacement) and the
controller host's clock. The closed schema is `ovl.rental-workload-health.v1`, with
`intent_sha256` selecting the watchdog intent, `pod_id`, `observed_epoch`,
`progress_epoch`, `exported_checkpoint_epoch` and boolean `complete`. The dispatcher
must recheck actual export evidence before setting `complete`; its export timestamp
must cover final progress. Malformed, stale or future health causes graceful stop,
not immediate deletion. The dispatcher and stop-delivery acknowledgement remain
pending; a timestamp or boolean cannot itself prove an export. The low-level
`observe()` policy and the controller's heartbeat precheck enforce one shared
watchdog requirement, not independent liveness evidence.

Provider account reads currently check image, GPU count, disk allocations and
creation time. Cloud/GPU type, exposed ports, SSH/Jupyter switches and resource
minimums are request-only fields and are explicitly reported as such. Actual GPU
and network/runtime admission must be checked separately before workload launch.
No network volume is currently permitted: every committed checkpoint must be
exported and verified off-pod before advancing. The fixed-cost forecast must include
those transfers and checkpoint delivery; this remains an execution requirement.
Residual network volumes are read from the provider and reported, not assumed
empty or deleted. An unrelated account resource conservatively stops this rental,
since its cost attribution/reservation assumptions then need reconciliation.

New CUDA13 rentals use `ovl.rental-controller-intent.v2`, which requires an exact
ordered `allowedCudaVersions` list in the creation payload. Historical v1 intents
remain replayable without alteration. The catalog availability query does not
constrain a separate creation request: diagnostic `5c4zcuwdyyk1y0` received CUDA12.8 /
driver570 and was explicitly terminated before numerical admission. The new field
is a provider request, not proof of actual driver compatibility; always observe
the selected device and driver before large setup transfers or numerical work.
See [RunPod's creation filter](https://github.com/runpod/docs/blob/main/sdks/graphql/manage-pods.mdx)
and [NVIDIA's CUDA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
