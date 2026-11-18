# Roadmap

## Done

- **July 2026: reproducibility study.** Same-GPU recomputation matches bit for bit; other
  architectures do not. A segmented replay audit cost 0.173 of training, against a theoretical
  0.15.
- **September 2026: data provenance.** The 2026-09-01 Wikipedia dump and OpenAssistant files
  were published with a signed source statement, together with the prepared token streams.
- **October 2026: the simple run.** A 23M model was trained on one exact pass over English
  Wikipedia, and all 37 segments were recomputed bit for bit. The data rebuild from the raw dump
  gives a byte-identical `preparation.json`. It also includes:
  - a hardware reproducibility map;
  - a verifiable exclusion fork (one article the run hadn't reached, retrained without it);
  - RFC 3161 timestamps, per boundary since 21 Oct;
  - a one-click segment check on RunPod.

  See [how it works](how-it-works.md).
- **Late October 2026: hardening `verify`.** Contributor fixes: `verify` fails when it ran no
  checks, clearer output for non-released weights, and CPU tests for `simple/ovl.py`. Since v1.0.1,
  a missing `anchor.json` is a failure unless `--no-anchor` is given explicitly. Since v1.0.2,
  `verify` also checks both timestamps on `anchor.json`, ties each chain entry's model digest to its
  weights, and accepts the released base model (#263, #267; see the peer evaluation in
  [`audit/`](../audit/README.md)).
- **31 October – 1 November 2026: cleanup (v1.1.0).** The old pipeline and the July study were
  removed from `main`; all the code is now in `simple/`, and the docs are five pages. The model has
  its own Hugging Face repository,
  [`AOSSIE/OpenVerifiableLLM-23M`](https://huggingface.co/AOSSIE/OpenVerifiableLLM-23M), and the
  exclusion fork is anchored.

## Next

- **Independent checks.** Get segments checked by people outside the project, using the
  [one-click check](verify.md#2-one-segment-one-click). This is the single most
  valuable next step for the release's credibility.
- **Cheaper checks.** Publish per-segment data slices, so that checking one segment doesn't need
  the full 22 GB of prepared data.
- **Per-article membership proofs.** For any Wikipedia article, show which segment trained on it,
  with a proof against the chain.
- **Near-duplicates in the exclusion fork.** The fork removed the 3I/ATLAS article as one document.
  Other articles that mention the comet stayed in, so the model may still have learned most of its
  facts. Count the training articles that mention it, and compare both models' loss on those
  passages. Then try excluding by topic (every range that mentions it), not by document. A topic
  exclusion is only as good as the filter that picks the ranges, so record the filter's code and
  version in the chain, and report its misses on a hand-checked sample. Compare against control
  articles about equally rare subjects, not just random articles of similar length, to separate
  what the removal did from its general knock-on effect on later training.
- **A static project site** that shows the record, the checks and the limits (in review: #264, #265).
- **Signed check results and random segment choice.** Today a check result is a self-reported line.
  Checkers would sign their result, and the segment would be picked from public randomness (for
  example a drand beacon round), so results can't be faked in bulk or aimed at honest segments.

## Open research questions

- **Checking across GPU architectures.** Bits currently match only on the training architecture.
  Integer or fixed-point training, or a reference path with defined rounding, could let other
  hardware check the same run.
- **Checking for less than the cost of training.** A full check costs as much as training. Proofs
  for some steps, or a checker protocol with attested random sampling, could lower that.

## Planned, pending funding

- **A bigger model** with the same method. Measured cost of training plus checking every segment
  at current throughput:

  | Model | Cost |
  |---|---|
  | 42M | about US$34 |
  | 70M | about US$52 |
  | 110M | about US$73 |

## Not planned

- A hosted demo app. The model runs locally in Ollama.
