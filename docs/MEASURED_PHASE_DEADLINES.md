# Measured phase deadlines

Recording and replay can have different measured checkpoint costs. Recording
exports new state bytes; exact replay can reuse an already retained object only
after its identity is checked. Both pilot measurements include their actual
checkpoint work. Neither measurement guarantees future throughput.

The monetary forecast remains conservative: charge both complete paths at the
slower observed rate, then add 25 percent. The aggregate operating limit and
protected reserve are unchanged.

Phase-time admission uses the authenticated measurement for each direction. For
each corpus, extrapolate every production update using the number of full-shape
pilot batches, round upward to milliseconds, add 25 percent, and round upward to
seconds. A final partial production batch receives the full-batch allowance.
Recording additionally reserves the entire configured publication deadline for
every public boundary. The caller still must fit registration, both selected
phase windows and terminal exports inside the original rental work deadline.

Each phase also has a frozen positive fixed-work reserve. It must be at least
the sum of its pilot setup measurements plus 25 percent, and must cover additional
production preparation such as checking the complete recorded chain before replay.
Measured setup is a minimum for that selection, not proof of a worst-case bound.
Exceeding the original phase window still stops the job.

The controller verifies every retained recording object before replay. Live replay
checkpoints must reuse all matching, rehashed objects; missing or corrupt objects
fail closed before a payload transfer. A cold transfer cannot silently replace the
measured replay path. This reuses storage bytes, never numerical computation:
replay still regenerates initialization and recomputes every update.

Registration construction receives a finite exposure allowance inside the original
rental reservation. It moves that allowance from fixed future costs into committed
exposure, preserving the total monetary envelope. This covers bounded elapsed
construction time without refreshing its selected epoch, extending any deadline,
or dropping the existing prior-exposure check.

This separates two forecasts; it grants no acceptance credit and changes no
numerical recipe. Complete same-host qualification, representative checkpoint
density, regenerated initialization, verified public precommitment, complete
coverage, continuous exact replay and strict parent identity remain required.
Changes apply only to future frozen selections. An existing workload cannot gain
time by changing its measurement, source or deadline.

The motivating [retained qualification and admission-refusal report](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/resolve/0357230f3cbc4d96d99167cb82dfdc1f90a63f24/technical-reports/closed-clean-phase-window-v27/closure.json)
records a successful pilot and initial-state regeneration followed by refusal
before production. That rental was terminated. Its evidence is not a completed
production pass or independent third-party verification.
