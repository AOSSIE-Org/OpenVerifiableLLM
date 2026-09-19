# Production candidate export and replay comparison

`ovl_pipeline.production_export` authenticates the production registration, source
commitment and complete progress prefix before reading safe checkpoints. It exports
both the base boundary and final conversational boundary. Each model has exactly
five files: `model.safetensors`, `config.json`, `preparation.json`,
`tokenizer-manifest.json` and `tokenizer.json`. Export does not restore optimizer or
CUDA RNG state. The tokenizer's actual preparation parent must match registration.

An export is labelled `EXPORTED_NOT_TRAINING_VERIFIED`. Signatures and self-consistent
weight hashes cannot establish the computation that produced the checkpoint.
The `replay-check` command performs the complete sequential numerical replay from
regenerated initialization, then compares both exported models and inference
configurations to that in-process result. It accepts no stored PASS report in place
of replay. This comparison alone does not reconstruct raw data, verify a final
public release, or establish generated-answer factual accuracy.

Run from the exact registered source with the locked environment. All arguments
below are concrete paths selected by the caller; policies must come from the
caller's trust configuration, not from an untrusted release.

```bash
python -m ovl_pipeline.production_export export \
  --packet PACKET --registration-bundle REGISTRATION_BUNDLE \
  --production-policy PRODUCTION_POLICY --source-policy SOURCE_POLICY \
  --source-checkout SOURCE_CHECKOUT --chain-directory CHAIN \
  --progress-directory PROGRESS --progress-policies PROGRESS_POLICIES \
  --prepared PREPARED --output FRESH_EXPORTS

python -m ovl_pipeline.production_export replay-check \
  --packet PACKET --registration-bundle REGISTRATION_BUNDLE \
  --production-policy PRODUCTION_POLICY --source-policy SOURCE_POLICY \
  --source-checkout SOURCE_CHECKOUT --chain-directory CHAIN \
  --progress-directory PROGRESS --progress-policies PROGRESS_POLICIES \
  --wikipedia-stream WIKIPEDIA_STREAM --conversation-stream CONVERSATION_STREAM \
  --exports FRESH_EXPORTS --output FRESH_REPLAY

python -m ovl_pipeline.production_export infer \
  --directory FRESH_EXPORTS/chat --registration PACKET/registration.json \
  --registration-sha256 CALLER_SELECTED_SHA256 --phase chat \
  --prompt 'Hello' --max-new-tokens 64
```

The numerical replay requires the audited GPU launcher and registered compatible
runtime described in [runtime auditing](RUNTIME_AUDIT.md). The commands above show
module arguments; they do not bypass that launch gate. Local inference is greedy,
FP32, CPU, one sequence, bounded to at most 256 new tokens, with an explicit
base/chat input template and retained-last-context policy. The registration hash
selects the recipe; the final release verifier must additionally select the trusted
model root. The standalone inference command is not a publication verifier.

The local tests use actual tiny CPU training, safe checkpoints, prepared inputs,
complete replay and inference, with explicit publisher/CUDA substitutes. They reject
altered weights, templates, tokenizer parents, extra files, symlinks, incomplete
replays and wrong boundary ancestry. They provide no actual production CUDA,
public-release or independent third-party verification credit.
