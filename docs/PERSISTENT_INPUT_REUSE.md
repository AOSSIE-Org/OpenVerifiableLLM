# Reusing prepared inputs on a quota-limited volume

`pod_fetch_prepared.py --reuse /volume/closed-inputs` selects an explicit,
caller-owned immutable input tree. Every selected file must be a read-only regular
file of the pinned size and full SHA-256 digest. The helper reads every byte on
every use and creates hardlinks into a fresh output tree on the same filesystem.
Missing, changed, writable or symbolic-link inputs fail; they do not trigger a
network fallback. A fresh download remains available by omitting `--reuse`.

Reuse assumes exclusive ownership and closure of all previous writers. Read-only
mode bits alone do not establish closure or revoke existing writable descriptors.
Consolidating old inputs is a separate operation: first verify exact identities,
retain the selection and recovery manifest off-volume, and preserve a verified
canonical copy. Never consolidate mutable journals or numerical output merely
because names or sizes match. Interrupted operations must preserve sole copies.

Cache activity uses `ovl.public-input-cache-read.v1`, distinct from network
response activity. It grants bounded liveness only. The result separately records
reused and downloaded bytes; reconstruction and training replay remain `NOT_RUN`.
Caching supplies neither transformation reconstruction nor numerical replay credit.

For a quota-limited network mount, select both `OVL_VOLUME_ROOT` and
`OVL_VOLUME_QUOTA_BYTES` in every pinned job environment. The root must cover the
entire tenant-charged volume, and the limit must come from the selected provider
resource, not `statvfs` of a shared storage pool. The worker counts each reachable
inode once, using the greater of its logical size, allocated size and 4 KiB. It
never follows symlinks. The same selection is checked across qualification and
initialization and carried into production and replay. Existing jobs without
these fields retain their historical ordinary-filesystem behavior.

The live census is an observation, not a storage reservation or an allocation
sandbox. It refreshes at most every 30 seconds and can lag concurrent writes; an
unlinked file held open can remain charged while absent from the census. Before
launch, establish exclusive resource ownership, reconcile open/deleted files and
check the complete remaining workload allocation bound against the quota. Include
all retained data, full record and replay states, simultaneous staging, package
inputs, journals and an explicit operating reserve. Recheck after qualification
using the observed footprint. Periodic free-space checks supplement this full
budget; they cannot replace it or guarantee containment of arbitrary writers.

Census checks retain the original deadline. A pending operator stop follows its
existing grace interval without starting another census. Off-pod guards remain
necessary if filesystem operations or the worker itself become unavailable.

Pre-workload recovery can retain a finite, explicitly selected set of small
receipts through `Health.retained_preflight`. Each selection binds the pod,
original watchdog intent, report names and an original recovery deadline.
Complete receipt bytes are rehashed off-pod and on journal adoption. Repeated
receipts cannot renew export age. Changed bytes, identities or deadlines fail.
These exports provide cost-health evidence only: they create no workload job,
process-exit record, completion event or model verification credit.

The recovery caller must authenticate actual remote reports, reconcile its one
mutation owner, retain inspection manifests before modifying redundant links and
verify all required recovery results before admitting work. It must release its
heartbeat and journal lease before the production coordinator adopts that same
journal. The original progress, export-age and shutdown guards remain active;
a heartbeat alone cannot renew useful progress or durable-export age.
