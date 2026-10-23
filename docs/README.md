# OpenVerifiableLLM Technical Documentation & Verification Contracts

This directory contains the formal technical specifications, verification contracts, and operational protocols governing the **OpenVerifiableLLM** verifiable pipeline.

OpenVerifiableLLM provides one-command cryptographic verification for small open model artifacts. To guarantee bit-exact reproducibility and falsifiable public provenance without trusted hardware or proprietary enclaves, the system decomposes execution into deterministic data preparation, bounded numerical workloads, signed public anchoring, and complete replay verification.

---

## Suggested Reading Order

Depending on your focus, we recommend the following reading paths:

### 1. New Contributors & Developers
1. [End-to-End Verifiable Wikipedia Plan](VERIFIABLE_WIKIPEDIA_PLAN.md) — The master architectural design and execution flow.
2. [Pipeline Formats](PIPELINE_FORMAT.md) — Specification of data schemas, manifests, and artifact layouts.
3. [Data Preparation Contract](DATA_PREPARATION.md) — How raw Wikipedia dumps are cleaned, chunked, and tokenized deterministically.
4. [Workload Execution](WORKLOAD_EXECUTION.md) — The execution model for training, replay, and validation runs.

### 2. Researchers & Independent Verifiers
1. [Complete Computation Verifier](COMPLETE_VERIFIER.md) — The contract for bit-for-bit model reconstruction and verification.
2. [Publication Boundary](PUBLICATION_BOUNDARY.md) — The strict partition between public evidence and private working material.
3. [Release Verification](RELEASE_VERIFICATION.md) — Dual-repository publication structure and verification contracts.
4. [Public Evidence Trace](PUBLIC_EVIDENCE_TRACE.md) — Current trace status and provenance requirements.

### 3. Infrastructure & Pod Operators
1. [External Watchdog](EXTERNAL_WATCHDOG.md) — Lifecycle enforcement, anti-orphaning guards, and lease safety.
2. [Resource Control Foundations](RESOURCE_CONTROL.md) — Run keys, process bindings, and numerical parent death.
3. [Production Lifetime](PRODUCTION_LIFETIME.md) — Guarded single-lifetime execution supervision.
4. [Public Anchoring](ANCHORING.md) — Cryptographic publisher identity and Sigstore/Git commit anchoring.

---

## Specification Catalog

The 37 specification documents are organized into six functional domains:

### 1. Architecture & Verification Foundations

Core specifications establishing the project's cryptographic trust model, artifact formats, and verification bounds.

| Document | Description |
|---|---|
| [VERIFIABLE_WIKIPEDIA_PLAN.md](VERIFIABLE_WIKIPEDIA_PLAN.md) | Master design for end-to-end Wikipedia training, replay, and verified publication. |
| [COMPLETE_VERIFIER.md](COMPLETE_VERIFIER.md) | Complete computation verifier reconstructing model weights from raw sources. |
| [PIPELINE_FORMAT.md](PIPELINE_FORMAT.md) | Schema definitions for pipeline manifests, records, and intermediate states. |
| [PUBLICATION_BOUNDARY.md](PUBLICATION_BOUNDARY.md) | Clear separation between verifiable public evidence and private operational data. |
| [PUBLIC_EVIDENCE_TRACE.md](PUBLIC_EVIDENCE_TRACE.md) | Public provenance requirements and status across development and production stages. |

### 2. Data Acquisition & Preparation

Protocols for downloading, verifying, extracting, and tokenizing large corpora without memory exhaustion.

| Document | Description |
|---|---|
| [DATA_PREPARATION.md](DATA_PREPARATION.md) | Full source-to-token preparation contract, streaming XML parsing, and tokenization driver. |
| [RAW_ARCHIVE.md](RAW_ARCHIVE.md) | Public raw Wikipedia archive transport, checksumming, and multi-chunk validation. |
| [PREPARED_STREAM_DOWNLOAD.md](PREPARED_STREAM_DOWNLOAD.md) | Byte-by-byte public HTTP stream download verification and Merkle tree validation. |
| [PREPARED_PUBLIC_RANGE_VERIFICATION.md](PREPARED_PUBLIC_RANGE_VERIFICATION.md) | Bounded HTTP Range request verification for massive dataset artifacts. |
| [PERSISTENT_INPUT_REUSE.md](PERSISTENT_INPUT_REUSE.md) | Immutable cached input reuse protocols under volume storage quotas. |

### 3. Training & Workload Execution

Specifications governing GPU training runs, step-by-step update tracking, and determinism pilots.

| Document | Description |
|---|---|
| [WORKLOAD_EXECUTION.md](WORKLOAD_EXECUTION.md) | Bounded workload containment, environment isolation, and execution monitoring. |
| [GPU_PILOT.md](GPU_PILOT.md) | Synthetic CUDA update census, optimizer bit-exactness, and divergence detection. |
| [SUSTAINED_PILOT_DISPATCH.md](SUSTAINED_PILOT_DISPATCH.md) | Sequential pilot dispatch within fixed-budget, pre-allocated GPU instances. |
| [MEASURED_PHASE_DEADLINES.md](MEASURED_PHASE_DEADLINES.md) | Timing models and measured deadlines separating recording from replay phases. |
| [RETAINED_PILOT_CYCLE.md](RETAINED_PILOT_CYCLE.md) | Validation of retained records, resume points, and bit-exact replay comparisons. |
| [PILOT_DELIVERY_WINDOW.md](PILOT_DELIVERY_WINDOW.md) | Allowances and timeouts for transferring pilot intermediate checkpoints. |

### 4. Runtime Security, Supervision & Safety

Safeguards ensuring compute instances fail closed, terminate on schedule, and protect against orphaned spend.

| Document | Description |
|---|---|
| [RESOURCE_CONTROL.md](RESOURCE_CONTROL.md) | Run keys, lease file descriptors, and numerical parent-death termination. |
| [EXTERNAL_WATCHDOG.md](EXTERNAL_WATCHDOG.md) | Independent watchdog enforcing strict RunPod rental bounds and shutdown guarantees. |
| [INITIALIZATION_SUPERVISION.md](INITIALIZATION_SUPERVISION.md) | Health probes validating hardware, drivers, and runtime dependencies before launch. |
| [POD_PUBLIC_RUNTIME.md](POD_PUBLIC_RUNTIME.md) | Safe container bootstrapping, diagnostic collection, and bulk export gating. |
| [RUNTIME_AUDIT.md](RUNTIME_AUDIT.md) | Hash-locked wheel payload verification and deterministic dependency audits. |
| [DURABLE_LIFECYCLE.md](DURABLE_LIFECYCLE.md) | Persistent state classification, error categorization, and safe lifecycle recovery. |
| [NETWORK_PLACEMENT.md](NETWORK_PLACEMENT.md) | Geographic and latency placement constraints for network storage volumes. |
| [TERMINAL_CAPACITY_REFUSAL.md](TERMINAL_CAPACITY_REFUSAL.md) | Handling provider supply constraints and terminal refusal states cleanly. |

### 5. Production Lifecycle & Checkpoints

Protocols for running multi-hour training runs, saving incremental states, and handling interruptions.

| Document | Description |
|---|---|
| [PRODUCTION_EXECUTION.md](PRODUCTION_EXECUTION.md) | Numerical trajectory coordination and public progress gate enforcement. |
| [PRODUCTION_LIFETIME.md](PRODUCTION_LIFETIME.md) | Single guarded production lifetime supervisor CLI and execution bounds. |
| [PRODUCTION_CHECKPOINT_RECOVERY.md](PRODUCTION_CHECKPOINT_RECOVERY.md) | Checkpoint retention, poll loops, and seamless recovery from dropped leases. |
| [PRODUCTION_EXPORT.md](PRODUCTION_EXPORT.md) | Export validation, registration checks, and candidate replay comparison. |
| [PRODUCTION_SCAN_OBSERVATION.md](PRODUCTION_SCAN_OBSERVATION.md) | Full filesystem scanning, file descriptor tracking, and supervisor polling. |
| [BOUNDED_CHECKPOINT_TRANSFERS.md](BOUNDED_CHECKPOINT_TRANSFERS.md) | Range-bounded chunked transfers for large safetensors files (>16 MiB). |

### 6. Evidence, Anchoring & Release

Publishing signed evidence, public git commitments, and verifying model release artifacts.

| Document | Description |
|---|---|
| [ANCHORING.md](ANCHORING.md) | Publisher key authorization, Sigstore attestation, and Git commit anchoring. |
| [PRODUCTION_ENDORSEMENTS.md](PRODUCTION_ENDORSEMENTS.md) | Multi-signature publisher endorsements and key validation contracts. |
| [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) | Verifying published Hugging Face models against recorded training manifests. |
| [EVIDENCE_PUBLICATION.md](EVIDENCE_PUBLICATION.md) | Immutable evidence archive upload to public repositories with atomic review gates. |
| [PERSISTENT_PUBLICATION.md](PERSISTENT_PUBLICATION.md) | Background systemd service managing reliable evidence uploads. |
| [PROGRESS_DISPATCH.md](PROGRESS_DISPATCH.md) | Handoff protocols for periodic training progress snapshots. |
| [PROGRESS_ENDORSEMENTS.md](PROGRESS_ENDORSEMENTS.md) | GitHub Actions OIDC workflow for signing incremental training progress. |
