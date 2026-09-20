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
# Guarded initialization cycle

`run_sustained_pilot.py` also accepts the distinct
`ovl.initialization-cycle-plan.v1` selection. It requires one initialization
record followed by one regeneration job, with independently selected recipe,
kernel, full stream, code, parameter count and discarded warmup count. Optional
runtime setup and complete public-input download stages use the existing guarded
transfer path.

`pod_initialization.py` runs `ovl_pipeline.initialization` through the selected
isolated runtime auditor. It accepts only `record` and `verify`, preserves the
original worker deadline and creates a fresh activity/audit directory. The
numerical initializer still requires a distinct process, regenerates state from
the recipe and compares every state tensor without restoring prover weights.

Every phase must fit its full measured work and export allowance. Initialization
does not use the pilot initial-state retention hook: its work plus complete final
export must fit the unchanged 1800-second export-age guard. A later admission
refuses insufficient remaining time; it never shortens or renews a phase.

Before regeneration, `initialization_parent.py` checks the complete retained
record tree and actual safe state against the external parent selection. The
worker independently rehashes every retained parent file. Before successful
completion, the dispatcher rechecks those bytes and the separate regeneration
report. Restart adopts the original descriptor and checks retained evidence;
it does not execute another numerical job.

Adoption also binds the exact ordered output-root list and retained terminal
record to the original job and endpoint. This includes interruption before final
health completion; rewriting a local result and its enclosing result hash cannot
relabel, omit or duplicate a declared output. Every retained local tree must be
covered by its complete inventory.

The retained consistency report does not itself execute regeneration, attest
hardware, authenticate a production publisher or grant production acceptance.
Development tests use actual tiny CPU states and file transfers with explicit
CUDA and process substitutes. Real CUDA initialization remains a separate gate.
