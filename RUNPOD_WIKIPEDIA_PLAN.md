# English Wikipedia training and publication plan

Prepared 2026-09-18. Status: superseded planning note; no paid resources launched.
User-reported RunPod balance: US$100. Scope: one GPU, model trained from random
initialization, one full English Wikipedia pass, conversation training, reproducibility
verification, and publication of downloadable model files.

**Use the
[end-to-end trust plan](docs/VERIFIABLE_WIKIPEDIA_PLAN.md) for execution.** The user now
requires public provenance and full data reconstruction plus complete training replay
before final verified publication, and explicitly permits exceeding 24 hours within
$100. The older provisional-release option, timing estimates and deadline gate below
are retained as planning history and must not override the new goal.

## Outcome and limits

- Publish a base model and a conversational derivative, with tokenizer, inference
  code, training configuration, corpus manifest, signed provenance, and replay reports.
- Start with approximately 23 million parameters. This is an experimental small
  assistant; completing training is not a claim of reliable assistant capability.
- Preserve the original 24-hour end-to-end target. It is a target, not a confirmed
  estimate. Adding credit creates retry room but does not make the pipeline faster.
- Separate time to first published artifact from time to a fully replay-verified
  release. A provisional release must state that full verification is pending.
- This plan does not provision infrastructure or publish anything by itself.

## 1. Finish the implementation locally before renting the long-running GPU

Existing code loads the corpus into memory and draws random batches with replacement.
That does not satisfy a full pass. The checkpoint chain also needs explicit data
position and precision support, and publication currently contains a toy replay recipe.

Implementation work:

| Area | Required change |
| --- | --- |
| Data preparation | Add resumable download, incremental XML extraction, deterministic tokenization, and hashed token shards. Never materialize the entire dump in RAM. |
| `src/dataset.py` | Add memory-mapped token loading and deterministic full-pass traversal, with exact target counts and persisted position. |
| `src/model.py`, `src/config.py` | Add tied embeddings, tokenizer/config identity, and the candidate configuration below. Preserve existing presets. |
| `src/chain.py` | Use a shared train/replay step supporting precision configuration, data cursor, scheduler, accumulation and complete training-state restoration. |
| Hashing and signing | Stream file hashes; bind dataset, code, environment and config to the run manifest. Use a new run-specific signing key without replacing existing keys. |
| Conversation training | Apply a recorded chat format and loss only to assistant tokens, then checkpoint the transition from the base model. |
| `src/publish.py`, `src/verifier.py` | Export the actual run recipe; require explicitly requested checks to pass rather than silently treating missing checks as verified. |
| Run supervision | Record progress, costs and ETA; checkpoint on controlled shutdown and support recovery without skipping data. |

Meaningful local tests: target coverage including final partial batch; no omissions
or duplicate target positions after resume; shard corruption rejection; fresh-process
checkpoint replay; assistant-loss masking; inference from an exported checkpoint;
required verification cannot report success when evidence is missing. Keep existing
regression tests passing. GPU-specific correctness is tested on the pilot.

## 2. Freeze the dataset and tokenizer

1. Select a completed dated official English Wikipedia pages-articles dump. Record
   exact URLs, release date and publisher checksums; verify downloads and compute
   SHA-256. Do not train from a moving `latest` URL.
2. Define the training corpus as current main-namespace, non-redirect article text,
   excluding XML wrappers, media, history, templates and discussion namespaces.
   Record extractor version and every exclusion category/count. Unexpected parse
   errors fail preparation rather than silently dropping articles.
3. Extract in stable order, retaining article/revision IDs and text hashes for
   provenance. Convert wikitext with a pinned deterministic extraction policy.
   This definition covers extracted article text, not every byte of the source XML.
4. Train a 32,000-entry byte-level BPE tokenizer on a deterministic bounded sample
   of that text; archive the sample-selection recipe and tokenizer files. Sampling
   here is only for tokenizer fitting. The language model still sees the full corpus.
5. Include BOS/EOS and chat markers before freezing the vocabulary. Verify tokenizer
   round trips against extracted UTF-8 text, with no lossy normalization or unknown
   token fallback. This is distinct from exact model-training replay.
6. Encode every eligible article, split long articles into windows without discarding
   tails, and write ordered uint16 token shards plus an article index. Record exact
   article, extracted-byte, token and prediction-target counts.
7. Use BOS for the first target and one-token input overlap at window boundaries;
   padding contributes zero loss. Each defined target position must appear once.
   Insert and document document separators. Context reuse is allowed; repeated loss
   targets are not. Do not drop the last incomplete batch.

The earlier approximately 26 GB figure describes compressed source data. It is not
the training-byte count. Final runtime is based on measured encoded target count.

## 3. Model and optimization candidates

Primary candidate: decoder-only transformer, 6 layers, hidden width 384, 6 attention
heads, tied input/output embeddings, vocabulary 32,000, approximately 23M parameters.
Start with context 512; benchmark 256 and 1024 if useful. A 256-token configuration
may limit conversational context and must be documented if selected. Use dropout 0
for the initial reproducibility-focused run.

Benchmark BF16 autocast with FP32 parameters/optimizer state first, then deterministic
FP32 if necessary. Test attention backends and compilation individually; keep a faster
option only after it passes the exact replay pilot. FP32 alone does not ensure replay.
Compare useful corpus targets/second, not padded tokens or a synthetic peak rate.

Use AdamW with an explicitly recorded starting learning rate, warmup and one-pass
decay schedule selected in the pilot. Benchmark microbatches and accumulation under
the fixed context, then freeze the configuration before production. Pilot updates
are discarded; production starts from its recorded random initialization.

An approximately 11M byte-token candidate is a fallback benchmark only if the primary
candidate fails the time/cost gate. Compare total corpus completion estimates because
the byte model has many more targets. Do not silently swap architectures or reduce
corpus coverage to make an ETA look better.

## 4. RunPod setup and pilot

- Check account access and live GPU availability/quotes before creating storage.
- Prefer one on-demand RTX 5090 32 GB; advertised price checked September 18 is
  $0.99/hour. Actual quote and region availability determine the deployment.
- Use an official PyTorch template with demonstrated RTX 5090 support. Record image
  digest, torch/CUDA/library versions, GPU and driver. Reuse its installed torch.
- Place persistent data and checkpoints on a network volume in the selected GPU's
  data center. Initially plan 250 GB; validate disk requirements and allow up to
  500 GB within the storage allowance. Stream extraction to avoid storing a giant
  intermediate XML file. Keep at least 20% free space.
- Prepare data locally if disk/CPU permit; otherwise use the quoted remote setup
  budget and include those paid hours. Do not pretend remote preparation is free.
- Keep long jobs detached from SSH, with logs on persistent storage and no public
  inference endpoint. Configure provider-side termination plus earlier graceful
  checkpoint shutdown; verify the guard exists before starting the long job.

Pilot acceptance, within a $10 setup/pilot allowance:

1. Warm up and measure representative real batches for at least 10 minutes, including
   checkpoint I/O. Report useful targets/sec, extracted bytes/sec, peak VRAM, and time
   spent loading/saving. Include compilation separately in the total ETA.
2. Compare two short fresh starts, then an uninterrupted run with a fresh-process
   checkpoint resume. Require exact model/buffer and relevant optimizer, scheduler,
   RNG, accumulation and data-position equality. Negative tampering must fail.
3. Measure or complete the actual corpus token count, then project total cost and
   end-to-end time. Include a 25% runtime margin, conversation training, full replay,
   export and storage. An early sample estimate is provisional, not the final gate.
4. Start production only if correctness passes and the forecast fits the spending
   guard. If the 24-hour forecast fails, report the actual forecast before the long
   run; extra balance is not permission to silently waive the deadline or exactness.

## 5. Train one complete pass, then the conversation stage

- Use stable sequential shard traversal with persisted offsets, no random sampling
  with replacement. Train every defined Wikipedia target once.
- Save approximately 32 signed training boundaries, choosing fixed step counts
  from the final token count. Also save recoverable checkpoints about every 30
  minutes if boundaries are farther apart. Preserve the initial state.
- On resume, restore optimizer, scheduler, RNG and data position. Preserve batch and
  accumulation semantics, including the final partial batch's loss normalization.
- Track consumed targets against the dataset manifest; fail completion if coverage
  does not match. Monitor finite loss, throughput, free disk, costs and ETA.
- Publishable base-model checkpoint is the end of this single Wikipedia pass.
- Conversation source: a pinned revision of `OpenAssistant/oasst1`, English only.
  Reconstruct valid conversation trees; exclude deleted/rejected messages and choose
  top-ranked assistant branches with deterministic tie breaking. Keep the official
  validation split separate and report resulting counts, not assumed 50k examples.
- Use one conversation epoch initially, with additional epochs up to three only if
  validation supports them and the time/cost gate permits. Preserve complete short
  dialogues; deterministically window long ones and document any excluded targets.
- Record a separate signed phase manifest linking the base checkpoint, conversation
  data, chat format, optimizer initialization, and final conversational checkpoint.

## 6. Verify and evaluate

Byte-exact verification concerns numerical state, not whether generated text copies
Wikipedia. Canonically hash named tensors and complete relevant training state;
serialized `.pt` container bytes need not be identical.

- Immediately replay a small selection of production segments to catch problems.
  Report exactly which segments passed; sampled checks do not establish full replay.
- For the fully verified release, replay both training phases sequentially from their
  original initial state, carrying replay state forward across all boundaries. Compare
  every recorded boundary and final state. Do not merely trust each supplied opening
  checkpoint independently. Reuse the same physical GPU/environment where possible.
- Record the scope: exactness demonstrated on this pinned stack. Do not claim
  cross-GPU or cross-version exactness, or universal proof of training honesty.
- Report Wikipedia training loss as training loss, not held-out performance, because
  the user requires every eligible article in training. Evaluate conversations on
  the untouched conversation validation split and a fixed set of qualitative prompts.
- Show representative successes and failures. Do not infer factual reliability
  from falling loss, checksum integrity, or a successful replay.

## 7. Publish, test the download, and release paid resources

Prepare a Hugging Face model repository under the user's selected account/repository.
The destination and available publication credentials remain to be resolved before
upload; local package creation does not depend on them.

Package base and conversation safetensors, tokenizer, architecture/config, usable
inference loader and chat example, model card, data/license attribution, environment
lock, actual training recipe, hashes, public signing material, and verification report.
Publish the required replay boundaries in a companion evidence repository if size
warrants it. Keep credentials and private signing keys out of all artifacts.

Download the published files into a clean environment, check hashes/signatures, load
the model and generate a response. Do not claim automatic Transformers or Ollama
compatibility without a working adapter/conversion and test.

After verified off-pod copies exist, terminate compute. Inventory remaining billable
storage; remove temporary data only after required artifacts are safely copied.
Continuous chatbot hosting is outside this budget and deliverable.

## Budget and time controls

| Allocation | US dollars |
| --- | ---: |
| Remote setup, preparation and pilot | 10 |
| Production training, conversation stage, evaluation and publication | 25 |
| Complete replay | 25 |
| Storage during execution | 5 |
| Unallocated retry/slowness reserve | 35 |
| Total available | 100 |

Treat $100 as a ceiling, not a spending target. Trigger a controlled checkpoint and
stop new compute at a projected cumulative $90, preserving $10 for export and remaining
storage. Track all created resources and actual quoted rates. Any reserved retry must
still fit the aggregate cap; do not run an unbounded restart loop or enable auto-top-up.

Planning ranges, not measurements: local implementation 6–12 hours; data preparation
2–6 hours; production Wikipedia pass 9–18 hours; conversation 0.5–2 hours; evaluation
and publication 1–3 hours. Full replay adds roughly the duration of both training
phases, plus comparison overhead. Approximately 19–41 hours to first publication
and 29–61 hours through full replay is the current rough sequential envelope.
Data preparation can overlap local implementation when feasible.

The earlier 9–18-hour Wikipedia estimate assumes 6–12B targets at about 185k targets/s
from an external 5090 report, not this implementation. Replace it with:

`training_hours = measured_total_targets / measured_sustained_targets_per_second / 3600`

Then add measured preparation/checkpoint/export time and the 25% forecast margin.
Both the original 24-hour goal and the $100 cap must be assessed using this result.

## Completion checklist

- [ ] Dated corpus, exact eligible-article count, token count, hashes and extraction policy frozen.
- [ ] Pilot passes fresh-start and fresh-process resume equality within the budget gate.
- [ ] One complete Wikipedia training pass; no missing or duplicate prediction targets.
- [ ] Conversation model trained and evaluated; base checkpoint retained.
- [ ] Full replay result reported honestly, including any failure or pending stage.
- [ ] Published files load and generate in a clean environment.
- [ ] Provenance, limitations, licenses and actual total cost recorded.
- [ ] Compute terminated and residual storage accounted for.

## Sources and execution references

- [RunPod advertised pricing](https://www.runpod.io/pricing)
- [Official English Wikipedia dumps](https://dumps.wikimedia.org/enwiki/)
- [OpenAssistant conversation dataset and license](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [External 5090 training comparison; not a measurement of this pipeline](https://github.com/Padraigobrien08/nanogpt-from-scratch/blob/main/docs/scaling.md)
- Local implementation starting point: OpenVerifiableLLM commit `03551dd`.
- RunPod workflow: official PyTorch batch-training pod, persistent volume, detached job,
  measured pilot, provider cost guard, verified artifact export, and teardown.
