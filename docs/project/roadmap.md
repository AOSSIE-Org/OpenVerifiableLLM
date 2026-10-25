# Roadmap

## Done

- **July 2026: reproducibility study.** Same-GPU recomputation matches bit for bit; other
  architectures do not. A segmented replay audit cost 0.173 of training, against a theoretical
  0.15.
- **September 2026: data provenance.** The 2026-09-01 Wikipedia dump and OpenAssistant files
  were published with a signed source statement, together with the prepared token streams.
- **October 2026: the simple run.** A 23M model was trained on one exact pass over English
  Wikipedia, and all 37 segments were recomputed bit for bit. The data rebuild from the raw dump
  gives byte-identical files. It also includes:
  - a hardware reproducibility map;
  - a verifiable unlearning fork;
  - RFC 3161 timestamps, per boundary since 21 Oct;
  - a one-click segment check on RunPod.

  See [the simple run](../design/simple_run.md).

## Next

- **Independent checks.** Get segments checked by people outside the project, using the
  [one-click check](../guides/check_a_segment.md#one-segment-one-click). This is the single most
  valuable next step for the release's credibility.
- **Cheaper checks.** Publish per-segment data slices, so that checking one segment doesn't need
  the full 22 GB of prepared data.
- **Per-article membership proofs.** For any Wikipedia article, show which segment trained on it,
  with a proof against the chain.

## Planned, pending funding

- **A bigger model** with the same method. Measured cost of training plus checking every segment
  at current throughput:

  | Model | Cost |
  |---|---|
  | 42M | about US$34 |
  | 70M | about US$52 |
  | 110M | about US$73 |

## Not planned

- A separate docs website or hosted demo app. The repository docs and the Hugging Face release
  are the documentation.
