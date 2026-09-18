# Bounded public runtime setup and selected bulk export

These operational tools do not certify training. The initial implementation and
ordinary advisory are in `project/evidence/public-runtime-transfer-v1/`. Full
production reconstruction/replay, public identity and numerical state checks remain
separate required gates.

The caller selects a hash-pinned configuration for `scripts/pod_public_setup.py`.
It binds the complete regular-file source archive inventory, public interpreter
archive, offline configuration, wheel location inventory and executable helpers.
The exact source builder uses USTAR; links, PAX metadata, bytecode, additional
members and file/ancestor collisions are refused. Source archives supply selected
bytes, not executable extraction instructions.

The setup process downloads selected public wheels from the closed HTTPS origin
list in `scripts/pod_fetch_runtime.py`. Each file is completely hashed before
promotion. At most two attempts are made for a transient network error, under one
original wall/monotonic deadline; changed bytes are never retried to success.
Failed partials and attempt receipts are retained. Receipt fields describe
operator-observed requested/final URLs, redirects, headers, counts and clocks.
They are not an independent network-history attestation.

Only then does `scripts/pod_runtime_setup.py` perform the existing full wheel and
installed-payload audits, public interpreter extraction/audit, offline hash-locked
installation and isolated runtime inspection. Installer temporary files are under
the selected runtime. The outer setup cross-checks both child receipts and their
parents before emitting PASS. Its selected config remains alongside the result.

Run the pinned entrypoint with the selected image's isolated bootstrap interpreter:

```sh
/usr/bin/python3 -I -S /selected/inputs/pod_public_setup.py \
  --config /selected/inputs/public-config.json --config-sha256 SELECTED_SHA256 \
  --inputs /selected/inputs --runtime /selected/runtime \
  --output /selected/setup-evidence --deadline ORIGINAL_JOB_DEADLINE_EPOCH
```

The original job/rental deadline remains authoritative. Entry is additionally
bounded to 300 seconds, of which at most 210 seconds are selected for downloads.
Earlier full offline local installation/audit measured 58.6507 seconds; that does
not prove the same elapsed time on a pod. A complete local public download attempt
failed on a read timeout after 48.687 seconds and stopped before installation.
The second remote rehearsal subsequently completed full acquisition, installation
and audit in 174 operator-observed seconds. This is one operational measurement,
not a guaranteed setup duration or a CUDA numerical result.

The persistent `pod_job_worker.py` supervisor must own the process group. Helpers
inherit that group so its timeout/exit cleanup reaches ordinary grandchildren.
Do not add `start_new_session` to installer children: that would escape this
existing cleanup boundary. The real local subprocess-timeout/grandchild regression
checks this composition. Neither it nor the external provider guard claims
hostile-process containment. Standalone callers own their subprocess cleanup.

Fresh-attempt refusal protects existing bytes. Never delete a failed source,
wheel or runtime tree to make a saved job rerun. Adopt the existing job, preserve
its outputs, and select new isolated attempt paths if a new execution is needed.
The finite coordinator never silently reruns a started job.

For at least eight selected files, `pod_job_client.export_tree` uses
`pod_bulk_export.receive`: one SSH stream concatenates only the caller-selected
regular file bytes. The receiver independently splits and hashes every file.
Complete remote inventories are checked before and after. Versioned exports
retain immutable content objects and reuse only fully rehashed matching objects.
The versioned exporter retains its threshold of 16 missing objects.
Interrupted raw streams/files remain preserved; automatic salvage of incomplete
bulk transfers is not implemented. A finite stage can make one fresh sibling
export attempt after an incomplete transfer; both attempts are retained. A
malformed completed receipt is never replaced by a retry.

Bulk reception reserves two full byte copies plus 20% filesystem headroom before
transfer and checks the original wall/monotonic deadline through splitting and
before completion. Finite planning budgets ten full hash passes. Local tests
show a 101-file complete export with three SSH operations; no remote throughput
speedup is claimed before measuring it. The inventory ceiling remains 100,000 files
and 16 MiB framing; production must prove its selected snapshots fit those bounds.

Export PASS means all selected peer bytes were retained and rehashed. It cannot
replace authenticated checkpoint ancestry, complete state comparison or full
numerical replay. The enclosing stage obtains its roots from the pinned job,
converts them into the independently selected SSH profile namespace, and retains
the job records too. SSH first-use host-key pins remain operator TOFU unless a
separate corroboration actually exists.

`scripts/pod_tiny_cuda_probe.py` runs an eight-update synthetic record, complete
fresh-process replay and a separate resume probe through the audited launcher.
The complete tiny probe is bounded to 240 seconds from entry, additionally capped
by its original job deadline and monotonic elapsed time. It does not combine
liveness sequences from different numerical process identities.
It does not supply sustained throughput, corpus coverage, a production model or
independent third-party verification.

Finite job observation uses `pod_observe.observe_many` to read caller-selected
metadata in one bounded exchange after separate process inspection. Each file is
opened once and its framing checked; this is not an atomic snapshot across files.
All present bytes are retained before canonical JSON parsing. Metadata reads do
not count as useful computation or renew progress. Each exchange is bounded by
both its short transport deadline and the unchanged external rental deadline.
The selected activity producer must write the canonical bounded JSON schema in
`runtime_activity.py`; malformed telemetry gets no progress credit. A durable
terminal observation with an absent runner and child permits export even if the
final mutable status or activity file is unusable. Unsafe output types still fail
regular-file retention checks.

A retained, exactly matching launch intent permits read-only reconciliation and
export after the compute deadline. It never permits another start. New work stays
subject to its original compute deadline, external guard, and any stop request.
This distinction lets a restarted coordinator preserve already completed work.
Both guards remain responsible for teardown even if observation or export fails.
Normal plans reserve export time before the requested provider deadline. Emergency
salvage during the external guard's grace interval is best effort; provider
availability after the requested deadline is not assumed or guaranteed.

The first remote rehearsal completed 75 of 76 wheels; the cuDNN wheel reached
508,559,360 of 553,099,438 bytes at the 180-second limit. The job failed before installation,
all 88 selected evidence files were rehashed off pod, and both guards confirmed
teardown. Its remaining prefix is exactly reconstructible from the retained
complete audited public wheel. This measured failure motivated 210 seconds for
downloads, leaving 90 seconds of the unchanged 300-second setup budget.
The second rehearsal completed setup but then stopped on a coordinator failure;
all 193 selected files were preserved and freshly verified after public download.
No active deadline was extended. CUDA and production acceptance remain pending.
