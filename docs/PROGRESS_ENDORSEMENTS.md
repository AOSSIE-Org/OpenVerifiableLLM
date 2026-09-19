# Public progress endorsement path

`anchor-progress.yml` signs one newly added request at the current branch HEAD.
Requests live under `project/progress-commitments/` and are named
`RUN_ID-ATTEMPT_ID-boundary-NNNNN.json`. They are append-only in the observed Git
ancestry; a deleted/re-added or modified request is refused. This is not a global
anti-equivocation guarantee or a replacement for branch protection.

A request selects the complete registration packet, its immutable public signature
archive and exact publisher policy; the signed run prefix; all preceding immutable
progress archives and policies; and the current immutable checkpoint archive.
The builder validates closed inventories and approved AOSSIE repository names.

The signing job performs these checks before requesting a signature:

1. Authenticate source and registration endorsements afresh and bind the actual
   code/dependency-lock bytes to the registered ancestor commit.
2. Verify every run-key signature and the exact primary-boundary schedule.
3. Force anonymous downloads of all prior progress statements/signatures and verify
   the complete prior public prefix under the selected exact policies.
4. Force anonymous downloads of all three current checkpoint files, rehash every
   byte, decode the safe complete state and compare its control with the signed
   run boundary.
5. Build an assertion binding registration, exact boundary, previous public
   assertion and immutable public checkpoint inventory; sign using the separate
   `anchor-progress.yml` OIDC identity and reverify the resulting prefix.

Checkpoint archives contain exactly `checkpoint.json`, `state.json` and
`state.safetensors`, under
`production-checkpoints/REGISTRATION_SHA256/boundary-NNNNN`. Registration bundles
use `production-anchors/REGISTRATION_SHA256/registration.sigstore.json`.
Progress archives contain exactly `statement.json` and `statement.sigstore.json`,
under `production-progress/REGISTRATION_SHA256/progress-NNNNN`. All revisions are
fixed 40-character HF commits; mutable branch names are refused. Transport caches
are not verification evidence: downloads use `token=False`, `force_download=True`
and complete content hashes before any state check.

The Actions artifact is temporary transfer staging. The external operating
controller must archive the new signature publicly, freshly download it, construct
its exact policy from independently selected source/statement identities and give
that verified public prefix to the paused recording process. This publication/
controller integration remains pending. The recording process reruns signature
checks; a saved PASS receipt alone cannot advance it.

Local tests cover actual Git ancestry, Ed25519 run signatures, real tiny safe-state
files and full downloaded byte checks with controlled transports. Publisher-positive
integration cases explicitly substitute the signature adapter; a separate test
uses the actual Sigstore adapter to reject fake bundles. No real production
progress request or signature has yet been created. Neither a signature nor a
consistent checkpoint proves its training history; the complete numerical replay,
fresh raw reconstruction and final public-download verification remain required.
