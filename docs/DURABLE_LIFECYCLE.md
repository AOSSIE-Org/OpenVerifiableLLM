# Durable lifecycle development

The synthetic lifecycle executes extraction and tokenization, random initialization,
Wikipedia and conversation training, raw reconstruction, continuous exact replay,
base/chat export, clean local copying, inference verification and closure.

Run from a checkout with the pipeline dependencies installed:

```sh
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -m ovl_pipeline.lifecycle_fixture run \
  --source tests/fixtures/pipeline --run artifacts/lifecycle-demo
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -m ovl_pipeline.lifecycle_fixture verify \
  --source tests/fixtures/pipeline --run artifacts/lifecycle-demo
```

The second command starts a fresh verifier process. It reconstructs the fixture
inputs and replays every update from regenerated initialization, then checks both
exports and inference. It never restores prover tensors into replay state.
The result remains `local-synthetic-fixture`: public anchoring and production
acceptance are `NOT_RUN`. A local copy is not a public download, and an operator
test is not independent third-party verification.

The run directory contains immutable stage outputs under `objects` and private
execution records, trust configuration, signing material and recovery files under
`private`. Never publish the run directory. Only explicitly inventoried and
reviewed payloads may be submitted through a publication adapter.

Run the same command after an interruption to recover the same identity. Source,
environment, operation and parent identities must still match. Completed outputs
are rechecked. Uncommitted local transformations are rebuilt in fresh scratch
directories, with interrupted bytes preserved. Recording resumes validated durable
boundaries; replay starts again from initialization. A missing journal alongside
old outputs is an error requiring evidence reconciliation, not a fresh run.

The execution journal records intent before an external mutation. If the response
is lost, recovery observes the original operation and checks its result. An empty
listing alone does not authorize repeating an uncertain creation. Only explicitly
classified transient reads receive bounded retries. Authentication, permission,
identity and integrity errors remain failures.

`lifecycle_guard` is a separate deadline supervisor with no creation authority.
It retains the original wall and monotonic deadline on restart, accounts for termination grace, and
does not disarm on an empty pre-creation listing. Its process must be independently
supervised before admitting a paid rental. Provider automatic shutdown is not
assumed. Local process tests are not evidence of provider automatic termination or
filesystem behavior under power loss.

`lifecycle_process` runs one selected module under a detached supervisor. Its
Linux supervisor requires Python 3.11 or later; the audited runtime uses the
selected Python 3.12 distribution. Other package entrypoints retain their own
runtime requirements. Unsupported supervisors are rejected before submission.
Its
request binds source, input bytes, output location and the original deadline.
The workload retains the exclusive lease through the audited launcher and its
numerical child. Linux parent-death signals stop numerical work if an enforcing
parent disappears; the independent provider guard still owns rental shutdown.
The binding is installed separately in each process and checked against the
parent's PID, boot identity and start time. See the
[Linux parent-death signal contract](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).

An audited workload uses `ovl_pipeline.runtime_launch`, with complete wheel and
interpreter origins, placing the audit in `OUTPUT/audit` and numerical artifacts
in `OUTPUT/result`. Its actual exit and exact bytes are operational evidence;
the scientific verifier must still check them. On recovery, a completed audited
target can supply its original exit receipt even when the supervisor receipt was
lost. A completed recording without that receipt remains uncertain. It is never
sent to the recording driver's incomplete-run resume path. Replay restarts from
initialization; no process receipt permits restoring prover state into replay.

Run the supervisor and audit parent in a separately trusted verifier environment
with the selected source and the `rfc8785` and `packaging` dependencies. Source
identity checks do not import Torch or other target numerical packages. The
audited child alone imports the separately installed numerical runtime.
`scripts/pod_verifier_setup.py` builds and audits that minimal environment from
the already selected public interpreter and bootstrap wheels. The synthetic
fixture also accepts `--output` for execution through this audited launch path.
Start the initial verifier with `-B -s -S -P`, a newly created
`-X pycache_prefix=...` directory, and an explicit `PYTHONPATH` containing only
the selected source and minimal verifier site-packages. Python 3.12 does not add
the virtual environment's site-packages under `-S`. Detached supervisors and
audit wrappers preserve these paths and create their own fresh cache prefixes.
The setup commands also propagate a fresh prefix to pip's delegated interpreter.
Both bootstrap setup commands accept an absolute `--deadline` and apply a shared
monotonic deadline to interpreter environment creation and package installation.
The default installation bound is one hour. Timed-out installers and their process
groups are killed; interrupted installation trees remain available for inspection.
See the [Python startup options](https://docs.python.org/3.12/using/cmdline.html).

Audited lifecycle receipts bind the original complete process request and source
digest. Recovery retains an observed deadline failure and the prior recording
resume selection through interrupted archival. Provider guard restarts retain
the last authenticated observation time; they cannot renew its recovery window.
The guard also accepts a trusted `stop_when` callback for execution-owner liveness.
An early stop closes creation admission permanently and remains selected while an
uncertain creation is being reconciled. Provider observations and billing remain
separate from any execution-owner heartbeat.

`lifecycle_runpod` supplies bounded provider inventory, creation and termination
adapters. REST v2 is used for reads and deletion; the existing GraphQL creation
contract is retained for its explicit `terminateAfter` field. `lifecycle_storage`
publishes exact inventories at operation-specific immutable prefixes, reconciles
lost commit responses and verifies anonymous downloads with cache sidecars outside
the payload. Transport checks do not replace publisher identity, signatures,
public anchoring, full scientific replay or publication privacy review.

These adapters are development components. Production orchestration, signed public
acknowledgements, CUDA qualification and production release acceptance require their
own integrated evidence before a paid or production run is admitted.
