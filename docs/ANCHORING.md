# Publisher identity and public anchoring

The development identity test runs only on pushes to the authorized implementation
branch, constructs its own statement from clean source, and signs it with GitHub
Actions OIDC. It cannot authorize preparation, training or a verified model release.
It accepts no caller-provided attestation text. The existing legacy model-publishing
workflow is separate and is not a production identity trusted by this pipeline.

The externally supplied publisher policy pins repository and organization numeric
IDs, exact source commit, branch, workflow, issuer, GitHub-hosted runner and statement
SHA-256. An artifact cannot select its own trusted issuer/root. Sigstore 4.5.0 verifies
signature, certificate, inclusion proof and signed checkpoint against production TUF
roots. Dependencies are hash-locked in `requirements/anchoring.lock`. An unavailable
required check fails; a saved PASS receipt is not an input to verification.

```
PYTHONPATH=src .venv/bin/python -m ovl_pipeline.anchoring \
  --statement statement.json --bundle statement.sigstore.json \
  --trust-policy independently-selected-policy.json
```

Obtain the expected policy from an approved source revision, check its digest out of
band, and select its exact statement/source identity before verification. Merely
copying a policy beside a model provides no independent trust selection. CI's own
self-check policy is labeled accordingly. The `--policy-origin` label records an
operator statement; no cryptographic proof of independence is implied.

PASS means these bytes were endorsed by the configured publisher and included in
the checked log state. It does not establish statement truth, production admission,
source reconstruction, training replay or answer factual accuracy. Cross-checkpoint
log consistency and independent witness observations are explicitly NOT_RUN.
Tree-relative proof index/root identify inclusion; a checkpoint envelope hash is
only that envelope's byte identity. TUF root-export digests are diagnostic snapshots
and can differ after legitimate trust-root updates.

Small immutable evidence is committed to this source PR and archived at fixed
revisions of the authorized AOSSIE evidence dataset. Actions artifacts are temporary
transport, not the sole public archive. Private operator progress is kept locally. Public evidence inventories and scoped
reports identify actual checks; no smoke test passes G01–G10. After repository
history maintenance, use the [current provenance status](../project/evidence/history-maintenance-v1/PROVENANCE_STATUS.md)
and independently select fresh publisher policies. Do not edit old signed payloads
or substitute a rewritten commit inside an old statement.

The forecast calculator in `ovl_pipeline.budget` uses exact monetary units, reserves
$10 beyond the $90 operating limit, requires at least ten-minute measurements, and
charges for both complete phases and full replay with a 25% runtime margin. It is
arithmetic only; provider deadlines, actual billing reconciliation, verified pilot
reports and complete target counts are additional required gates before rental.
