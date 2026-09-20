# Astra proposal before advisory

Use maintained sigstore 4.5.0 with production TUF bootstrap and online refresh;
verify exact Actions workflow SAN, issuer, repository URI, commit digest and ref.
Bind exact canonical statement bytes to an independently supplied digest. Never
interpret this as training/statement truth; return explicitly scoped evidence.
A branch-push-only smoke constructs its own statement from clean tracked source;
it accepts no caller-controlled attestation text. This tests credentials and
external anchoring before any production authorization is implemented.

Falsifiable assumptions: real Actions certs carry required v2 repository/digest and
legacy ref extensions; the SDK checks included signed checkpoint/inclusion; stale
or substituted statement/proof/identity cannot return PASS. Tests + live smoke
will decide these claims. Unavailability must fail closed. No production gate
is currently opened by this module. No raw production prep until source commitment.

Lane: native Opus ordinary read-only challenge of pinned packet; Astra sole writer
and integration owner. Concurrent local full-data planning and original raw download
use disjoint resources. Packet/result under project/evidence/anchoring and managed
advisory store. Finish on evidence-bearing objections/no supported objection; actual
checks and disposition before integration. 1800s/64000 output allowance selected.
