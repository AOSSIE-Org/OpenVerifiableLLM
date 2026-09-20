# Off-pod public progress handoff

`scripts/publish_progress_boundary.py` handles one paused production boundary.
It authenticates the separately selected source/registration policies, verifies the
signed prefix and full safe checkpoint bytes, publishes the checkpoint to the
existing AOSSIE evidence repository, and downloads every file at a fixed revision.

It then commits exactly one append-only request in a dedicated clone and pushes the
existing source branch without force. The operator constructs the expected statement
and policy from the selected checkpoint archive and actual pushed commit before
reading signing outputs. The exact first Actions attempt must succeed. Temporary
signing outputs are verified, archived publicly, freshly downloaded, and verified
again. Only then is `ack.json` written. That acknowledgement supplies the exact
anchor directory and external policy to deliver to the paused recorder.

The external workload dispatcher must copy the two verified anchor files first and
publish the policy list atomically last. `production_record` independently verifies
the whole prefix before another update. This script does not provision, terminate,
transfer from a pod, deliver that acknowledgement or assert training verification.
Those operations still require the live workload/controller integration.

`scripts/pod_checkpoint_handoff.py` now supplies the bounded SSH snapshot and
acknowledgement operations. Its CLI authenticates source/registration policies and
code before any remote command. Select a profile for the already adopted pod,
private local SSH key and pinned known-hosts file. The checkpoint-handoff profile's remote root must
be the recorder's output directory, `/workspace/ovllm/OWNED_NAME`; use its
`anchors/` subdirectory and root `external-progress-policies.json` as recorder
arguments. Neither SSH host-key TOFU nor provider endpoint metadata attests GPU
hardware or training.

Use a separate job-control profile, such as `/workspace/ovllm/OWNED_NAME-jobs`,
for the worker script and launch journals. Its host, pod ID and host-key selection
must match the handoff profile. Uploading `tools/` or `jobs/` into the recorder
output would create that directory before `production_record` can enforce its
fresh-output requirement. The handoff profile is used only after the recorder
creates its own output. The existing local tests do not yet validate this complete
production launch arrangement; it is a production-integration prerequisite.

Run `snapshot` with `--profile`, `--key`, `--known-hosts`, `--packet`,
`--registration-bundle`, `--production-policy`, `--source-policy`,
`--source-checkout`, `--output FRESH_SNAPSHOT`, and `--deadline ABSOLUTE_EPOCH`.
It downloads chain/waiting metadata, verifies the exact run-signed schedule,
transfers all three checkpoint files against that signed inventory, decodes the
safe state/control, and re-reads both remote markers before recording export.
This receipt establishes a complete local copy, not public availability or replay.
Missing, changed and interrupted copies remain preserved without a PASS receipt.

After the publisher above returns its verified acknowledgement, run `deliver`
with those same identity/transport options, a fresh output directory,
`--snapshot-directory FRESH_SNAPSHOT`, `--ack PUBLICATION/ack.json` and
`--progress-policies SEPARATELY_SELECTED_POLICIES`. It recomputes state integrity
and every public signature under the external policies, checks the pod is still
paused at that boundary and rejects any policy rollback. Immutable anchor files
are copied first; the policy list is installed atomically last. An interruption
can leave anchor files but cannot authorize advancement with a partial prefix.
The recorder independently verifies again. A retry verifies and adopts matching
immutable bytes; it never overwrites an existing different anchor. Delivery does
not repeat the publisher's public network downloads or prove recorder advancement.

The persistent stage runner, cost-health integration and stop/export lifecycle
remain prerequisites to a paid workload. These CLI operations do not provision
resources, renew deadlines or mark the whole workload complete.

```bash
PYTHONPATH=src .venv/bin/python scripts/publish_progress_boundary.py \
  --packet REGISTRATION_PACKET --registration-bundle REGISTRATION_BUNDLE \
  --production-policy OPERATOR_PRODUCTION_POLICY --source-policy OPERATOR_SOURCE_POLICY \
  --source-checkout PINNED_CHECKOUT --config DISPATCH_CONFIG \
  --chain-directory OFFPOD_RECORD_SNAPSHOT --previous-directory VERIFIED_PREVIOUS_ANCHORS \
  --previous-policies OPERATOR_PREVIOUS_POLICIES \
  --output PUBLICATION_ROOT/boundary-00000 --deadline CHECKPOINT_STOP_EPOCH
```

`DISPATCH_CONFIG` is canonical JSON with `schema: ovl.progress-dispatch.v1`, the
already published `registration_request` and `registration_anchor`. The previous
policy list is empty for boundary zero. Later outputs use the same publication
root with consecutive boundary names; previous acknowledgements bind their archives.
Credentials stay in the normal local HF/GitHub stores; no private run key is used.

A lease prevents concurrent publication of the same boundary. Uncertain HF writes
are reconciled read-only and the selected archive revision stays fixed even when
unrelated repository commits arrive. A possibly successful Git push is never blindly
repeated: recovery fetches and proves the selected commit's public ancestry. An
incomplete local commit window requires inspection of preserved evidence; it never
authorizes another training update. Completed acknowledgements are retained and
must be verified/adopted rather than republished. No failure extends a rental.

The recorder CLI is available through `python -m ovl_pipeline.production_record
--help` and the audited launcher. It takes an externally selected registration SHA,
owner-only key directory, full streams, output, anchor directory, external progress
policy file and checkpoint deadline. The seed is never a command-line value.

Optional `OVL_ACTIVITY_FILE=/absolute/owned/output/activity.json` emits a bounded
operator observation after actually completed GPU updates, at most once every
30 seconds. It does not change the numerical control or RNG state. The cost
supervisor must assign its own observation time to new sequence numbers; remote
timestamps, process liveness and this telemetry cannot prove a checkpoint export,
training correctness or an extension of the fixed rental deadline.

Local tests use actual tiny CPU safe checkpoints and full byte transfers through a
fake HF provider, plus explicit publisher/Actions substitutes. A real local Git
repository tests a successful push with a deliberately lost caller response and
read-only recovery. These tests do not supply live public signing, CUDA, provider
or independent-verification acceptance credit.

Git commit and push commands may use up to 600 seconds to complete mandatory
publication hooks. Other commands retain their 120-second per-command limit.
Registration and progress publication clip each command to the remaining original
publication deadline, with a monotonic cap that prevents a backward wall-clock
adjustment from renewing the command window. A command completing at or after that
limit is rejected. These are internal command allowances: no privacy hook, public
download, signature, acknowledgement or original rental deadline is waived. Hook
failures remain failures, and uncertain writes retain the reconciliation rules
above. The monotonic cap is per invocation and covers wrapped commands, not all archive
operations or a persisted elapsed-time clock across recovery. Recovery retains the
original epoch deadline. The enclosing publication supervisor still owns
whole-process termination.
