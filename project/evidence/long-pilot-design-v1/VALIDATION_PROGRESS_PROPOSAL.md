# Bounded next correction: full-stream startup liveness

Design only, no code/acceptance yet. The actual unchanged local CPU Wikipedia
validator took210.047389s and checked all7232582rows/7118121139targets, including
full file hashes, SQLite identity uniqueness, masks, EOS and final Merkle root.
This plus the complete audited launcher leaves little margin under the unchanged
300s useful-progress guard. A slower remote CPU can exceed it without stalling.
No further rental should depend on unmeasured favorable startup timing.

Preserve frozen preparation modules while clean reconstruction runs. Candidate
implementation outside those files: a scoped wrapper around the unchanged stream
validator, observing completed index-row prefixes after its generator resumes
from each yield (the previous row's checks have run). Restore the original iterator
in finally; refuse concurrent/reentrant instrumentation and unexpected input paths.
Emit a separate truthful preflight observation only after full input-file hashing
has passed; emit final completion only after validate_stream returns and verifies
all counts/root. No skipped checks, copied validation logic, fake numerical steps,
remote timestamps or heartbeat credits. Test failures on a row and the final root:
the failing row cannot earn prefix credit; full prefix does not mean final PASS.

A dedicated pilot health consumer must independently bind the selected stream
manifest digest/document total to the job. Prefix counts are finite, monotonic,
non-repeating and bound to the same process identity/sequence as subsequent actual
numerical activity. Polling cannot renew progress or durable export age. Keep the
existing finite-v3 dispatcher admission unchanged; a separately versioned sustained
pilot dispatcher must preserve fixed rental,300s progress,1800s durable-export and
external deadline+120s limits.

Full-data pilot costs include complete public stream transfer, every full validation
and runtime audit, initial/middle/final safe states, full replay and candidate resume.
The candidate's148MBinitial and333MBupdated states require measured retention budgets.
Consider retaining immutable completed boundary directories while the pilot runs,
with complete safe-state checks, so a long final export does not exceed1800s export
age. Mutable logs remain excluded from such interim snapshots; complete terminal
retention is still mandatory. Do not credit byte/object reuse until verified. Derive
replay's record digest only from complete retained record bytes before launch.

Current independent full-corpus batch16/context512 census:1120354Wikipedia updates,
293conversation updates; training not run. Neither tiny CUDA timing nor bounded
batch-tokenization timing supplies sustained forecast credit. Required next empirical
checks remain>=600s representative pilot per stream, complete fresh replay, actual
candidate resume, and a cost comparison of configurations meeting exactness/corpus
requirements before production recipe/init/precommitment.
