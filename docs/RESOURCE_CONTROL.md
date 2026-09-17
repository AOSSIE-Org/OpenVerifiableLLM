# Run keys and resource-control foundations

These modules are implemented and locally tested. No production run key has been
created, no paid resource has been provisioned, and no live provider guard or
supervisor has been admitted. A local policy result is not provider-side evidence.

## Private run keys

Create one key in a fresh directory under an existing owner-controlled parent:

```sh
python -m ovl_pipeline.run_key create --directory PRIVATE_DIRECTORY --run-id RUN_ID
python -m ovl_pipeline.run_key check --directory PRIVATE_DIRECTORY --run-id RUN_ID \
  --expected-public-key EXTERNALLY_PINNED_PUBLIC_KEY
```

The output contains only the public descriptor and its canonical digest. The
private directory is mode0700; `seed.key` is a raw32-byte Ed25519 seed in an
owner-owned mode0600 regular file. Symlink components, hard-linked seed files,
wrong permissions, incorrect lengths, nonregular files, mismatched public identity
and wrong run IDs fail closed. The seed is never printed or included in reports.
The standard Ed25519 test vector and persistence/adversarial cases are covered.

Existing directories, including incomplete creations, are never overwritten or
regenerated automatically. Preserve them for explicit recovery. After creation,
make an owner-only **copy** on durable backup storage and check that copy against
the same externally retained public pin before any remote work. A hard link is
not a backup and is refused. Never put either secret copy in a public archive,
Git commit, model repository, job log or advisory packet. Only `public.json`
belongs in the public production registration. Production authorization requires
that registration's verified public anchor; possession of a key grants none.

On interruption, adopt the existing key using the independently known registration
public key. Do not accept a replacement public key merely because it accompanies
the downloaded secret or run artifacts. Lost private material requires a declared
new attempt/key and new public authorization, preserving the old evidence.

## Rental deadline policy

`ovl_pipeline.supervision.rental_plan()` consumes a closed
`ovl.rental-budget-input.v1` object. It takes actual prior project spend,
outstanding commitments, remaining mandatory-work reservations, a per-rental
allowance, a quoted **all-in upper hourly rate**, the quote digest, the current
epoch and bounded runtime/checkpoint/billing margins. All monetary values are
nonnegative decimal USD strings converted to integer microdollars.

The lifetime starts before provisioning, so setup is charged. Available funds are
the smaller of the rental allowance and the $90 operating budget after prior
spend, commitments and reservations. The plan rounds affordable time down, charges
billing slack and starts checkpointing before the provider termination deadline.
The final $10 stays protected. If even the checkpoint/billing margins do not fit,
planning fails. The result always says provider guard and execution admission
`NOT_RUN`.

`observe()` checks fresh normalized observations supplied by a controller. Only
pod IDs tied to project creation evidence may be targeted. It reports unrelated
active pod IDs separately. Multiple project GPUs, stale observations, changed or
unverified provider deadlines, missing recent progress/checkpoints, higher rates,
regressed reservations, low balance or the operating/deadline threshold request
`CHECKPOINT_AND_STOP`. This is a decision, not a provider mutation or proof that
shutdown happened. All-in observations must cover associated storage charges;
account-wide unrelated charges must not be mislabeled as project spend.

The remaining adapter must authenticate observations directly from RunPod,
reconcile billing and all resources across attempts, confirm provider-side
termination scheduling and preserve artifacts outside ephemeral pod storage.
The [official GraphQL schema](https://graphql-spec.runpod.io/) advertises
`stopAfter` and `terminateAfter` input fields, but their authenticated behavior and
readback are unverified here. Do not provision until a supported hard guard can
be set and checked. Local policy tests cannot substitute for that check.

Run the implemented read-only credential/schema/account probe with a fresh receipt:

```sh
python3 scripts/provider_preflight.py --output local-provider-observation.json
```

It reads `RUNPOD_API_KEY` or the standard local RunPod TOML configuration, sends
only fixed queries to the official HTTPS endpoint, refuses redirects, and keeps
credentials and raw errors out of reports. It reports balance, account-wide rate,
unattributed resource IDs and available deadline fields. It never creates resources,
changes auto-pay, validates termination behavior or grants execution admission.
Account-wide observations are not project spend. Missing credentials, partial/error
responses and unsupported shapes fail with a nonzero exit and a scoped receipt.
The empty default CLI config file is not evidence of authentication.

## Durable controller journal

`Journal(directory).lease()` acquires a nonblocking OS file lock. It never steals
a lease based on a stale PID; process death releases the lock. Each event is
written to a private draft, fsynced and atomically linked without overwrite to its
canonical event filename, followed by a directory fsync. Events bind the previous
event digest. A restarted controller reads retained intents before doing anything
else. Unpublished drafts are preserved and ignored; a killed writer cannot leave
a partial canonical event hiding earlier intents. Corrupt committed events, gaps,
symlinks and broken ancestry fail closed without truncating evidence. Real child
process death/restart tests cover death both before and after event publication.

A creation intent must be durable **before** the API request. If the response is
lost, reconcile that intent against provider state and adopt the resource; do not
issue a duplicate request merely because the process restarted. An actual
controller/adapter must implement this reconciliation; the journal itself does
not create, adopt, stop or delete resources.

Events are local operator observations, not externally anchored history. Preserve
immutable copies/public receipts at milestones; keep mutable `goal_state.json`
separate. Store only deliberately selected nonsecret fields, never complete API
responses containing credentials. A positive local decision never supplies
training verification or independent third-party acceptance.

## Operational native-deadline probe

`scripts/probe_provider_deadline.py` is a single-attempt empty-pod test, not a
production supervisor or admission gate. The authenticated API accepts native
`terminateAfter`, but currently rejects reading it back on `Pod`. A bounded
behavior test is therefore needed before drawing any operational conclusion. The
first test reserves$0.25 (projected conservative ceiling$0.075), uses one quoted
RTX2000Ada at$0.24/hour, and assumes an all-in upper bound$0.30/hour. Four GB of
container disk costs$0.10/GB/month under the [provider pricing documentation](https://docs.runpod.io/pods/pricing).
No volume, model weights, user data or credentials are placed on the pod.

A pinned minimal Ubuntu image sleeps until the requested native ten-minute deadline.
A systemd user service survives client disconnect and restarts against the same
creation journal; lingering is enabled. A caller fallback terminates by deadline+120s,
or earlier on observation/configuration/storage faults. That fallback is local,
not native provider evidence. Creation is never retried after a recorded intent;
a lost response is reconciled using its unique pre-request identity. Two actual
absence observations are required. Billing and residual storage need separate
reconciliation. API outages can still prevent caller cleanup.

The executable accepts only the fixed original probe journal directory. Its
`--dry-run` exercises fresh account/quote/clock/budget checks with no mutation.
A timing match can report only `OBSERVED_TERMINATION_IN_DEADLINE_WINDOW`; alternate
provider termination causes remain possible. It cannot authorize a later production
pod, prove scheduler state, or mark G10 passed. Full training and throughput rentals
remain closed until their own required guards and other trust gates are established.

Provider monetary observations retain their full precision. `policy_amount` floors
available balances and rounds costs upward to six decimals before integer-microdollar
budget policy. Never silently round a cost down or available funds up.
