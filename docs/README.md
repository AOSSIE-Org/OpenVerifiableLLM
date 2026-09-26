# OpenVerifiableLLM Documentation Index

This directory contains technical specifications, architecture blueprints, verification contracts, and operational protocols for the **OpenVerifiableLLM** verifiable training and release pipeline.

---

## Suggested Reading Order

If you are new to the codebase or looking to understand the verifiable training lifecycle, we recommend reading the documents in this order:

1. [VERIFIABLE_WIKIPEDIA_PLAN.md](VERIFIABLE_WIKIPEDIA_PLAN.md) — Master plan and end-to-end trust architecture for verifiable training from English Wikipedia dumps.
2. [COMPLETE_VERIFIER.md](COMPLETE_VERIFIER.md) — Specification of the verification engine (`production_verify`) that reconstructs execution and checks bit-exact claims.
3. [PIPELINE_FORMAT.md](PIPELINE_FORMAT.md) — Schema, canonical encoding, and file format specifications for the pipeline and synthetic fixtures.
4. [DATA_PREPARATION.md](DATA_PREPARATION.md) — Source extraction, tokenization, shard chunking, and deterministic dataset preparation.
5. [WORKLOAD_EXECUTION.md](WORKLOAD_EXECUTION.md) & [GPU_PILOT.md](GPU_PILOT.md) — Orchestrating bounded workloads and pilot training on GPU infrastructure.
6. [ANCHORING.md](ANCHORING.md) & [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) — Cryptographic signing, OIDC publisher identity, and release gates.

---

## Document Map by Functional Area

### 1. Architecture & Verification Foundations

Core designs that define what constitutes a verifiable model, the falsifiability contract, and the global verification engine.

| Document | Description |
|---|---|
| [VERIFIABLE_WIKIPEDIA_PLAN.md](VERIFIABLE_WIKIPEDIA_PLAN.md) | End-to-end design, phase budgets, and execution plan for verifiable English Wikipedia training and publication. |
| [COMPLETE_VERIFIER.md](COMPLETE_VERIFIER.md) | Specification of the complete verifier (`ovl_pipeline.production_verify`) that reconstructs training and validates parameter equality. |
| [PIPELINE_FORMAT.md](PIPELINE_FORMAT.md) | Canonical data formats, manifest schemas, and fixtures used across the verification pipeline. |
| [PUBLIC_EVIDENCE_TRACE.md](PUBLIC_EVIDENCE_TRACE.md) | Status of public evidence traces, trust boundaries, and criteria for valid verification claims. |

---

### 2. Data Acquisition & Preparation

Protocols for downloading, verifying, and preprocessing monolithic Wikipedia archives into deterministic training shards.

| Document | Description |
|---|---|
| [DATA_PREPARATION.md](DATA_PREPARATION.md) | Source-to-token transformation orchestrator, tokenization pipeline, and canonical dataset shard generation. |
| [RAW_ARCHIVE.md](RAW_ARCHIVE.md) | Transport and integrity requirements for publishing raw Wikipedia dump archives to public mirrors. |
| [PREPARED_STREAM_DOWNLOAD.md](PREPARED_STREAM_DOWNLOAD.md) | Streaming verifier protocol that validates hash integrity of public datasets chunk by chunk. |
| [PREPARED_PUBLIC_RANGE_VERIFICATION.md](PREPARED_PUBLIC_RANGE_VERIFICATION.md) | Range-bounded HTTP verification for large datasets to prevent timeouts and memory overruns. |

---

### 3. Training & Workload Execution

Specifications governing deterministic training runs, GPU pilot tests, and time-budgeted execution.

| Document | Description |
|---|---|
| [WORKLOAD_EXECUTION.md](WORKLOAD_EXECUTION.md) | Bounded workload orchestrator, SSH/process runners, and local execution contracts. |
| [GPU_PILOT.md](GPU_PILOT.md) | Procedures for launching and testing synthetic training cycles and determinism pilots on remote GPUs (RunPod). |
| [SUSTAINED_PILOT_DISPATCH.md](SUSTAINED_PILOT_DISPATCH.md) | Multi-step dispatching of consecutive training pilots within a single instance rental. |
| [PILOT_DELIVERY_WINDOW.md](PILOT_DELIVERY_WINDOW.md) | Checkpoint copy deadlines and delivery window constraints. |
| [MEASURED_PHASE_DEADLINES.md](MEASURED_PHASE_DEADLINES.md) | Phase timing measurements and deadline allocation for training recording vs. replay runs. |
| [RETAINED_PILOT_CYCLE.md](RETAINED_PILOT_CYCLE.md) | Verification of retained pilot checkpoints, full replay traces, and resume integrity. |

---

### 4. Runtime Security, Supervision & Safety

Safeguards against runaway cloud costs, memory exhaustion, untrusted dependencies, and hardware failures.

| Document | Description |
|---|---|
| [RESOURCE_CONTROL.md](RESOURCE_CONTROL.md) | Run keys, hardware affinity, and budget constraints to prevent resource overruns. |
| [EXTERNAL_WATCHDOG.md](EXTERNAL_WATCHDOG.md) | Lifetime supervision and watchdog daemon to prevent orphaned rented instances. |
| [INITIALIZATION_SUPERVISION.md](INITIALIZATION_SUPERVISION.md) | Health probes and supervision checks during model initialization and warmup. |
| [POD_PUBLIC_RUNTIME.md](POD_PUBLIC_RUNTIME.md) | Clean-room environment setup on rented pods and selective export sanitization. |
| [RUNTIME_AUDIT.md](RUNTIME_AUDIT.md) | Locked wheel audits and package payload SHA-256 verification. |
| [TERMINAL_CAPACITY_REFUSAL.md](TERMINAL_CAPACITY_REFUSAL.md) | Protocol for gracefully handling and recording cloud provider GPU capacity refusals (`SUPPLY_CONSTRAINT`). |
| [NETWORK_PLACEMENT.md](NETWORK_PLACEMENT.md) | Cloud datacenter/region placement constraints for latency and reproducibility. |

---

### 5. Production Lifecycle & Checkpoints

Lifecycle states, checkpoint pruning, and failover recovery for long-running training runs.

| Document | Description |
|---|---|
| [PRODUCTION_EXECUTION.md](PRODUCTION_EXECUTION.md) | Numerical execution rules, progress gates, and deterministic training drivers. |
| [PRODUCTION_LIFETIME.md](PRODUCTION_LIFETIME.md) | Complete guarded lifecycle of an authorized production run from start to finish. |
| [PRODUCTION_CHECKPOINT_RECOVERY.md](PRODUCTION_CHECKPOINT_RECOVERY.md) | Retention, pruning, and failover recovery for intermediate training checkpoints. |
| [BOUNDED_CHECKPOINT_TRANSFERS.md](BOUNDED_CHECKPOINT_TRANSFERS.md) | Chunked streaming rules for uploading and downloading model weights larger than 16 MiB. |
| [PRODUCTION_SCAN_OBSERVATION.md](PRODUCTION_SCAN_OBSERVATION.md) | Stream logging and startup scans for validating honest execution. |

---

### 6. Evidence, Anchoring & Model Release

Cryptographic provenance, Sigstore identity anchoring, transparency logs, and public export bundles.

| Document | Description |
|---|---|
| [ANCHORING.md](ANCHORING.md) | Sigstore publisher identity anchoring, OIDC proofs, and public commit transparency. |
| [PRODUCTION_ENDORSEMENTS.md](PRODUCTION_ENDORSEMENTS.md) | Publisher endorsements, cryptographic signatures, and model authority profiles. |
| [PROGRESS_ENDORSEMENTS.md](PROGRESS_ENDORSEMENTS.md) | GitHub Actions workflow signatures for incremental training progress receipts. |
| [PROGRESS_DISPATCH.md](PROGRESS_DISPATCH.md) | Dispatching immutable progress boundaries and intermediate receipts to public storage. |
| [EVIDENCE_PUBLICATION.md](EVIDENCE_PUBLICATION.md) | Archive transport for cryptographic proofs, run manifests, and replay logs. |
| [PERSISTENT_PUBLICATION.md](PERSISTENT_PUBLICATION.md) | Local persistent service configuration for long-running evidence dispatch. |
| [PUBLICATION_BOUNDARY.md](PUBLICATION_BOUNDARY.md) | Strict separation rules between public technical evidence and private operator context. |
| [PRODUCTION_EXPORT.md](PRODUCTION_EXPORT.md) | Packaging verified checkpoints into Hugging Face repository and Ollama build artifacts. |
| [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) | Final verification criteria and red/green decision logic for public model release candidates. |

---

## Contributing to Documentation

When updating or adding specifications:
- Maintain clear distinction between public verifiable artifacts and private runtime state (see [PUBLICATION_BOUNDARY.md](PUBLICATION_BOUNDARY.md)).
- If introducing new schema formats or manifest fields, update [PIPELINE_FORMAT.md](PIPELINE_FORMAT.md).
- Keep this index updated whenever new specification documents are added.
