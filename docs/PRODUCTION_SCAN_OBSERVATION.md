# Complete startup scans and supervision

Production initialization, recording and replay keep every complete stream
validator and independent coverage/cursor census. With `OVL_ACTIVITY_FILE` set,
`production_observation.py` runs the original function body with a private globals
mapping that observes row iteration and nested validation. It does not modify the
original modules or preparation code.

An observation counts a row only after its consumer resumes the iterator. Final
completion is emitted only after the original function returns, including its
final root and count checks. Nested validation runs before the outer census;
concurrent scans and reentry during row iteration are rejected. At most sixteen
scan passes may be observed in one process. Without the activity setting, the
original functions are called directly.

`ProductionHealth` binds observations to the independently selected registration,
stream roots and document totals, the original job retention contract, and one
process incarnation. Only advancing checked prefixes or final scan checks renew
useful progress. A new empty pass, duplicate observation or repeated sequence
cannot renew it. Numerical activity must retain the same process identity and
advance the sequence; subsequent scan observations are rejected.

These are operator supervision observations, not training verification or hardware
attestation. They never grant export or job completion credit, extend a phase or
rental deadline, or replace the full numerical replay. The provider deadline,
external watchdog, budget and complete retention requirements remain unchanged.

Local tests exercise actual tiny CPU record, interrupted recording/resume and
complete replay, with explicit publisher and CUDA substitutes. They also exercise
altered roots, counts, process identities, sequences, pass limits, final-check
failures and journal adoption. The same-process resume test resets the observer
explicitly; it is not evidence of a fresh operating-system process. Actual full
production dispatch and its initialization health integration remain pending.
