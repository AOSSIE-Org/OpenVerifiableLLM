# Bounded immutable checkpoint transfers

Selected files larger than 16 MiB are read in ordered ranges of at most 64 MiB.
The small-read threshold remains 16 MiB, so medium-sized files retain ranged
recovery and the payload-inactivity check. Each connection
has a 90-second payload-inactivity ceiling, with at most two failed-range retries for the complete
file. Every connection remains inside the original absolute and monotonic transfer
deadline. These are transport recovery limits, not extensions to checkpoint,
workload, rental or billing deadlines.

Only received payload renews the inactivity allowance. Stderr, process liveness
and progress callbacks cannot renew it. Productive ranges may take longer than
90 seconds if they finish within the original whole-file deadline. Waiting for
process exit also remains bounded. Synchronous callbacks and filesystem work
cannot grant timely-delivery credit after crossing the parent deadline.

Only explicitly classified transient transport failures with known byte counts
and completed child cleanup may retry. Authentication, host identity, selection,
framing, integrity and unknown failures stop recovery. Overlapping bytes from
failed attempts must agree. Failed bytes and receipts remain unverified private
evidence; only complete length and SHA-256 verification can install a download.
Uploads and workload starts are not retried by this mechanism.

Small immutable reads may retry twice after positively classified transient
failures only when both transport counters and the actual retained file show zero
payload. Backoffs of two and four seconds remain inside the original wall and
monotonic deadlines. Each retry rechecks the same peer, authentication files and
selected path, size and hash. Failed-attempt receipts and bounded diagnostics are
retained privately; success receipts exclude those diagnostics. Small and ranged
reads pass the original monotonic ceiling directly into each connection, so a
wall-clock rollback at stream entry cannot reconstruct a longer allowance.
Partial payload, failed cleanup, insufficient backoff time or exhaustion cannot reset this allowance through
a fresh snapshot. A successful retry still requires the complete size/hash check;
it supplies no numerical verification credit.

Progress reports logical file offsets to the durable health journal. Repeated
prefixes cannot earn additional progress after a retry or controller restart.
Attempt receipts separately count received application payload, including repeated
bytes; these are neither SSH wire measurements nor provider billing records.

A fresh checkpoint or terminal snapshot requires a preserved transient-failure
classification bound to the exact profile, root, selected inventory, bounds and
original deadline. It is allowed only when no payload arrived in the failing
operation. Missing classifications, partial payload, failed cleanup and exhausted
range retries cannot reset the recovery allowance. Completed objects still receive
the usual inventory, safe-state, control and parent checks before acknowledgement.

An exhausted development-stage export has a separate, private retention-only
path after the owned worker is observed terminal. Only classified transient range
exhaustion with completed transport cleanup is eligible. Its fixed intent binds
the failed job, source worker, peer, failure, selected roots, byte allowance and
deadline. The deadline is capped by the original stage export reserve, first
failure's shutdown allowance, controller stop and external rental deadline.
Logical and uncached-byte allowances remain separately bounded by the selection.
Each root has one backup attempt; restarting cannot reset an incomplete attempt.
Reverified complete objects may be reused. Recovered files must agree with all
preserved overlapping immutable observations, including failed range prefixes.
Previously selected metadata and completed small downloads remain binding too.
The first terminal observation is immutable, and a subsequent backup refusal is
preserved across restart. A completion-shaped export receipt alone cannot prove
that its final deadline check returned successfully. The enclosing coordinator
can adopt an interrupted backup between completed roots, using its original
phase identity and deadline without re-entering qualification.
Original failures and partials remain intact. This permits cost-safe teardown;
qualification stays failed, normal stage re-entry is refused and no successor
may use the failed dispatcher as a parent.

Range transfers can help when a long-lived connection degrades, but add connection
overhead and can perform worse with slow connection startup. A connection taking
95 seconds to start can fit a longer whole-file deadline while failing this range
policy. Reconnection cannot cure globally insufficient bandwidth. Local synthetic
tests cover these unfavorable cases as well as recovery from a connection-lifetime
stall. They do not establish performance improvement on a GPU provider.

The prospective 64 MiB policy reduces connection count for large checkpoints but
increases buffering and the bytes repeated or preserved after a failed range.
It retains the same file-wide retry count and original deadlines. A same-endpoint
comparison with complete payload verification and subsequent mandatory sustained
record/replay qualification must establish a useful total-cost improvement before
this candidate is selected for production. Source tests supply no throughput credit.

Verification checks the original deadline before granting retention or export
health. Blocking filesystem operations cannot provide a hard real-time guarantee;
late retained bytes are not evidence of timely delivery. The external rental
watchdog remains necessary. Local path and authentication-file checks assume a
trusted local controller filesystem and do not prove resistance to concurrent
privileged pathname replacement.

Focused checks:

```sh
PYTHONPATH=src:scripts TOKENIZERS_PARALLELISM=false python -m pytest -q \
  tests/test_checkpoint_range_recovery.py tests/test_pod_transfer.py \
  tests/test_payload_inactivity.py tests/test_failed_stage_retention.py \
  tests/test_failed_stage_integration.py tests/test_failure_retention_boundaries.py \
  tests/test_pod_versioned_export.py tests/test_pilot_checkpoint_delivery.py \
  tests/test_production_live_retention.py tests/test_workload_stage.py \
  tests/test_workload_health.py tests/test_reconcile_checkpoint_delivery.py
```

These checks validate operational recovery only. Full data reconstruction,
regenerated initialization, complete sequential replay, public commitments and
release download verification remain separate mandatory acceptance gates.
