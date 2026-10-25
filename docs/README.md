# Docs

This directory contains the project's documentation, from what the project is trying to do down to
how to check a training segment yourself.

If you're new to OpenVerifiableLLM, start with the [README](../README.md).

## Design

The [`design/` directory](design/README.md) explains how verifiable training works and what the
released run measured: [the simple run](design/simple_run.md).

## Project

The [`project/` directory](project/README.md) covers the [goals](project/goals.md),
[roadmap](project/roadmap.md) and [FAQ](project/faq.md).

## Guides

The [`guides/` directory](guides/README.md) has step-by-step instructions:
[check a training segment](guides/check_a_segment.md) and
[the `ovllm verify` artifact verifier](guides/ovllm_verify.md).

## Images

Logo and figures, with their sources: [`images/`](images/snippets.md).

## Legacy (frozen)

These pages describe the earlier production pipeline (`src/ovl_pipeline/`). It is frozen: kept for
reference, not used by the simple run apart from data preparation and a few library files. Some
published evidence links to these exact paths, so they stay where they are.

- [Publisher identity and public anchoring](ANCHORING.md)
- [Bounded immutable checkpoint transfers](BOUNDED_CHECKPOINT_TRANSFERS.md)
- [Complete computation verification](COMPLETE_VERIFIER.md)
- [Full-data preparation contract and driver](DATA_PREPARATION.md)
- [Durable lifecycle development](DURABLE_LIFECYCLE.md)
- [Immutable evidence publication transport](EVIDENCE_PUBLICATION.md)
- [Revised RunPod lifetime guard](EXTERNAL_WATCHDOG.md)
- [GPU development pilot and complete update census](GPU_PILOT.md)
- [Initialization supervision](INITIALIZATION_SUPERVISION.md)
- [Measured phase deadlines](MEASURED_PHASE_DEADLINES.md)
- [Network placement requests](NETWORK_PLACEMENT.md)
- [Reusing prepared inputs on a quota-limited volume](PERSISTENT_INPUT_REUSE.md)
- [Persistent checkpoint publication](PERSISTENT_PUBLICATION.md)
- [Pilot checkpoint delivery windows](PILOT_DELIVERY_WINDOW.md)
- [Version 1 development pipeline formats](PIPELINE_FORMAT.md)
- [Bounded public runtime setup and selected bulk export](POD_PUBLIC_RUNTIME.md)
- [Complete public prepared-data verification through bounded responses](PREPARED_PUBLIC_RANGE_VERIFICATION.md)
- [Complete prepared-data public response verification](PREPARED_STREAM_DOWNLOAD.md)
- [Production checkpoint retention and handoff recovery](PRODUCTION_CHECKPOINT_RECOVERY.md)
- [Production registration endorsement profile](PRODUCTION_ENDORSEMENTS.md)
- [Production numerical execution and public progress gates](PRODUCTION_EXECUTION.md)
- [Production candidate export and replay comparison](PRODUCTION_EXPORT.md)
- [One guarded production lifetime](PRODUCTION_LIFETIME.md)
- [Complete startup scans and supervision](PRODUCTION_SCAN_OBSERVATION.md)
- [Off-pod public progress handoff](PROGRESS_DISPATCH.md)
- [Public progress endorsement path](PROGRESS_ENDORSEMENTS.md)
- [Public evidence and private working material](PUBLICATION_BOUNDARY.md)
- [Public evidence status](PUBLIC_EVIDENCE_TRACE.md)
- [Public raw archive transport](RAW_ARCHIVE.md)
- [Public model releases and verification scope](RELEASE_VERIFICATION.md)
- [Run keys and resource-control foundations](RESOURCE_CONTROL.md)
- [Check retained record, full replay and resume states](RETAINED_PILOT_CYCLE.md)
- [Locked wheel payload audit](RUNTIME_AUDIT.md)
- [Sustained development pilots within a fixed rental](SUSTAINED_PILOT_DISPATCH.md)
- [Recovery from a complete capacity refusal](TERMINAL_CAPACITY_REFUSAL.md)
- [End-to-end verifiable Wikipedia training and publication](VERIFIABLE_WIKIPEDIA_PLAN.md)
- [Bounded workload execution components](WORKLOAD_EXECUTION.md)
