# Production registration endorsement profile

This interface authenticates publisher endorsements and checks relationships among
operator reports. It does **not** perform raw reconstruction, GPU initialization,
training replay or resource admission. A successful endorsement is not G01–G10
completion. The production execution/release path is still under implementation.

The verifier operator must select two policies separately from downloaded evidence:

- Source: `ovl.publisher-policy.v2`, exact approved `anchor-pipeline.yml` identity,
  source-signing revision and source statement digest.
- Production: the same closed policy schema, exact approved
  `anchor-production.yml` identity, production-signing revision and registration
  digest. The policy also fixes issuer, repository/owner numeric IDs, branch,
  hosted runner and Sigstore production TUF roots.

Never take a downloaded policy as the consumer's trust decision. The CI-generated
policies are retained as observations of CI's choices. The `--policy-origin` value
is a caller label, not cryptographic proof of independent policy selection.

The public packet contains exactly these regular files:

```text
registration.json
source-statement.json
source-statement.sigstore.json
preparation.json
initial-record.json
initial-verification.json
wikipedia-pilot-record.json
wikipedia-pilot-replay.json
conversation-pilot-record.json
conversation-pilot-replay.json
```

The registration binds the exact source-signature bundle and all parent roots, full-stream censuses, exact recipe and
kernel, numerical environment, immutable container, GPU dependency lock, code,
regenerated initialization, both sustained pilot/full-replay reports, complete
cost forecast and run public key. The packet holds reports/manifests; their large
prepared files, initial checkpoint and pilot checkpoints require their own public
archives and actual verification. Report consistency alone does not establish
that any reported computation took place.

After obtaining the packet and its registration signature, use the implemented CLI:

```bash
PYTHONPATH=src .venv/bin/python -m ovl_pipeline.production_anchoring \
  --packet /path/to/downloaded/packet \
  --registration-bundle /path/to/registration.sigstore.json \
  --production-policy /path/to/operator-selected-production-policy.json \
  --source-policy /path/to/operator-selected-source-policy.json \
  --source-checkout /path/to/trusted/source-checkout
```

The CLI verifies both signatures, exact certificate identities/revisions and
transparency inclusion anew. It rejects saved PASS strings as signature evidence,
missing/extra packet files, symlinks, oversized JSON, unsupported policies and
inconsistent parents. Failures exit nonzero. Success is explicitly
`publisher-endorsements-and-report-parent-consistency-only`; reconstruction,
training replay, provider guard and production admission remain `NOT_RUN`.
Without `--source-checkout`, code/dependency-lock binding is also `NOT_RUN`;
with it, the verifier recomputes the Git/source/lock checks. Installed runtime,
container identity, fixed-cost evidence and actual train/validation membership
remain `NOT_RUN` in this endorsement profile.
The Sigstore environment must satisfy `requirements/anchoring.lock` (also included
in the preparation environment); an unavailable verifier/root refresh fails closed.

## Publisher workflow

Upload a content-addressed packet to a new
`production-registration/<sha256>` prefix in an approved AOSSIE evidence dataset.
Pin the HF revision. The signing request uses `ovl.production-signing-request.v1`
with `registration_sha256`, `packet` (`repo`, `revision`, `prefix`, sorted exact
`inventory` of path/bytes/SHA-256 entries) and `source_policy`. Requests are stored
as `project/production-commitments/<run_id>-<attempt_id>.json`.

Push the request commit as the HEAD of its own push; adding it in an earlier
commit of a multi-commit push does not trigger signing and requires a new attempt.
A single-parent push commit may add exactly one new request. Changes, deletion,
reintroduction or multiple requests are rejected. Unrelated pushes do not re-sign
old registrations. This is an observed-ancestry constraint, not global protection
against history rewriting. The branch was observed unprotected on2026-09-18;
the owner prohibits force-pushing. Consumers must pin exact revisions/digests,
retain public log receipts, and must not trust the moving branch name as a root.
The job downloads all packet bytes without authentication,
checks the public file set and hashes, verifies the source signature, checks
parent consistency, validates the frozen source contract, and checks the declared
code and GPU lock against the signing checkout and an ancestor code commit.
The earlier code commit can precede the request commit so pilots can run before
the final registration is endorsed. Any changed executable module changes the
code root and must be reflected in the actual pilot/initialization evidence.

CI then signs the exact registration bytes with ambient GitHub OIDC and rechecks
both endorsements. Its output is an operator endorsement with checked parents,
not a hosted-runner reproduction of the reported GPU work. Archive all output at
an immutable public revision and verify a clean download; expiring Actions
artifacts are insufficient as the sole archive. No production signing request or
live production endorsement has been issued as of this implementation checkpoint.

Preparation currently runs locally under the same declared Python3.12.14/Linux/x86_64
preparation lock that CI checks. Matching this fingerprint is not host attestation.
Caller-owned evidence directories must not have a concurrent untrusted writer.
Failed Actions outputs are deliberately retained; an artifact or bundle alone does
not establish successful workflow completion or computations claimed in reports.
