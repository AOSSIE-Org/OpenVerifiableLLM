# Bounded immutable checkpoint transfers

Large selected files are read in ordered ranges of at most 16 MiB. Each connection
has a 90-second ceiling, with at most two failed-range retries for the complete
file. Every connection remains inside the original absolute and monotonic transfer
deadline. These are transport recovery limits, not extensions to checkpoint,
workload, rental or billing deadlines.

Only explicitly classified transient transport failures with known byte counts
and completed child cleanup may retry. Authentication, host identity, selection,
framing, integrity and unknown failures stop recovery. Overlapping bytes from
failed attempts must agree. Failed bytes and receipts remain unverified private
evidence; only complete length and SHA-256 verification can install a download.
Uploads and workload starts are not retried by this mechanism.

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

Range transfers can help when a long-lived connection degrades, but add connection
overhead and can perform worse with slow connection startup. A connection taking
95 seconds to start can fit a longer whole-file deadline while failing this range
policy. Reconnection cannot cure globally insufficient bandwidth. Local synthetic
tests cover these unfavorable cases as well as recovery from a connection-lifetime
stall. They do not establish performance improvement on a GPU provider.

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
  tests/test_pod_versioned_export.py tests/test_pilot_checkpoint_delivery.py \
  tests/test_production_live_retention.py tests/test_workload_stage.py \
  tests/test_workload_health.py tests/test_reconcile_checkpoint_delivery.py
```

These checks validate operational recovery only. Full data reconstruction,
regenerated initialization, complete sequential replay, public commitments and
release download verification remain separate mandatory acceptance gates.
