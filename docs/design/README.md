# Design

How the verifiable training method works, with the measured results behind each design choice.

- [The simple run](simple_run.md): the method (hash-linked segments, exact coverage, determinism,
  timestamps), results, hardware scope, verifiable unlearning, cost, timeline, things to avoid and
  lessons learned.

The [July 2026 reproducibility study](../guides/ovllm_verify.md#reproducibility-matrix) measured
when "same code + same seed" really gives the same model. Its results are the basis for the
determinism settings and the same-GPU requirement.
