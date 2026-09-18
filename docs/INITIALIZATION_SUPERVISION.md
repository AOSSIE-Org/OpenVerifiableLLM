# Initialization supervision

`scripts/initialization_health.py` consumes the initializer's existing
`ovl.runtime-production-scan.v1` observations before a production registration
exists. Its `InitializationHealth` class uses the existing watchdog intent,
journal lease and fixed rental clock. Callers independently select a binding for
each initializer job:

```json
{
  "schema": "ovl.initialization-validation-binding.v1",
  "stream_sha256": "<externally selected complete stream digest>",
  "documents": 7232582,
  "action": "record"
}
```

The example document count belongs to the current original Wikipedia preparation;
callers must use their selected manifest. `action` is `record` or `verify` and is
bound to the selected job through the journal. The enclosing dispatcher still
needs to bind the actual command and all its inputs. Its job kind is `pilot`:
initialization warmup discards its weights and does not authorize production.

Only pass one of `stream-validation` is accepted. The journal retains the raw
scan event, including its operation and pass index. Validation reuses the pilot
checks for exact stream identity, document counts, advancing checked prefixes,
process incarnation and sequence. Adoption repeats these checks. The numerical
warmup must retain the same process and advance its sequence. Polling can miss
the final scan snapshot; the actual initializer still runs every validation check
before warmup. Supervision telemetry does not replace that validation.

An empty scan, repeated observation or unchanged prefix cannot renew useful
progress. No scan grants checkpoint-export or job-completion credit. Rental,
watchdog, work and export deadlines remain unchanged. This helper does not launch
initialization, compare fresh processes, verify a public commitment or complete
G03. The production dispatcher must retain complete safe initial state and the
actual fresh-process regeneration report before any production registration.

Local tests use actual tiny prepared files and discarded CPU warmup with an
explicit CUDA substitute. Process-transition observations in those tests are
synthetic and labeled. Adversarial checks cover changed streams, pass indices,
counts, process identities, sequences, historical bindings and forged journal
cost decisions. They provide development evidence, not a CUDA qualification.
