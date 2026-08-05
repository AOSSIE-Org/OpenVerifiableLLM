# OpenVerifiableLLM

**One-command verification for small open model artifacts.**

OpenVerifiableLLM is an AOSSIE project for making model releases independently
checkable. The goal is simple: a stranger should be able to pull a published
model, run one verifier command, and see whether the weights, manifest, replay
claim, and publisher provenance all check out.

This repository contains both the research harness that tests training
reproducibility and the Phase B verifier/publish path that turns those proofs
into a usable artifact.

## Current Status

The project now has two connected layers:

- **Reproducibility experiments:** deterministic training, hardware/precision
  sweeps, safetensors artifacts, tensor hashes, Merkle chunking, and segmented
  replay audits.
- **Verifier and publish loop:** `ovllm verify` checks a local or Hugging Face
  model directory, recomputes hashes, validates Merkle metadata, optionally runs
  replay, and verifies Sigstore/model-transparency provenance.

The intended midterm story is:

```bash
ovllm verify <publisher>/<model>
```

If the artifact is intact and properly signed, the verifier prints green checks
and ends with:

```text
VERDICT: GREEN
```

## What `ovllm verify` Checks

Given a local model directory or Hugging Face model reference, the verifier:

1. Resolves the model reference.
2. Loads `ovllm_manifest.json`.
3. Finds the `.safetensors` weights.
4. Recomputes the raw artifact SHA-256.
5. Rebuilds the Merkle tree over weight chunks.
6. Recomputes the tensor-level safetensors hash.
7. Optionally runs a structured segment-replay audit if the manifest includes one.
8. Verifies the Sigstore/model-transparency bundle as a red/green provenance check.

Missing Sigstore provenance is **red by default**. For local development only,
use `--allow-unsigned` to turn that check into a skip.

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
ovllm --help
```

The editable install exposes the `ovllm` command used throughout this README.

Run the verifier against a prepared local directory:

```bash
ovllm verify path/to/model-dir
```

Remote Hugging Face references download into `.ovllm-cache/huggingface` by
default to avoid permission issues in the global Hugging Face cache. Override
with `--cache-dir <path>` or `OVLLM_HF_CACHE_DIR`.

If remote verification reports `signature present, but manifest lacks
sigstore_identity/provider`, the uploaded directory was not the GitHub
Actions-signed artifact. Re-run the **Publish Verified Model** workflow and
publish that signed output so the manifest includes the expected Sigstore
identity metadata.

For local unsigned smoke tests (skipping local retraining):

```bash
ovllm verify path/to/model-dir --allow-unsigned --skip-replay
```

For full end-to-end training verification (including local retraining and parameter hash matching):

```bash
ovllm verify path/to/model-dir --allow-unsigned
```

This will automatically execute the local CPU retraining pipeline defined in the manifest's `segment_replay` block, then verify that your locally trained model matches the uploaded model's parameter hash bit-for-bit.

Run tests:

```bash
python -m unittest discover -s tests
```

Run the CPU demo arc:

```bash
python demo.py
```

Run the segmented replay/falsifiability smoke test:

```bash
cd src
python reproducibility.py
cd ..
```

For GPU demo commands and the full matrix, see [RUNBOOK.md](RUNBOOK.md).

## Publish Loop

Prepare a publishable model directory from the real trained safetensors checkpoint:

```bash
ovllm prepare-publish \
  --weights artifacts/gpt10m_shakespeare_fp32_deton_s99.safetensors \
  --out dist/gpt10m-shakespeare \
  --name gpt10m-shakespeare
```

This writes:

- `model.safetensors` or the original safetensors filename
- `ovllm_manifest.json`
- `README.md` model card
- `Modelfile` for an Ollama build path

Rerun `prepare-publish` whenever the model-card or manifest template changes so
the publish directory contains the current generated metadata.

Signing for published artifacts is performed by the **Publish Verified Model**
GitHub Actions workflow, not by a local terminal. The workflow signs with GitHub
OIDC so the Sigstore identity is tied to this repository/workflow instead of a
personal local browser session.

Local signing is disabled by default to prevent developers from accidentally signing
with their personal accounts. If you need to test signing locally, set the environment
variable `OVLLM_ALLOW_LOCAL_SIGNING=true` (or `$env:OVLLM_ALLOW_LOCAL_SIGNING="true"` in PowerShell).

Run the workflow with:

```text
model_name: gpt10m-shakespeare
hf_repo_id: <user-or-org>/gpt10m-shakespeare
publish_to_hf: true
```

It signs with this expected identity shape:

```text
https://github.com/<owner>/<repo>/.github/workflows/publish-verified-model.yml@<git-ref>
```

and this provider:

```text
https://token.actions.githubusercontent.com
```

The workflow verifies the signed directory without `--allow-unsigned`, uploads it
as a GitHub Actions artifact, and can optionally upload it to Hugging Face when
`HF_TOKEN` is configured as a repository secret.

Manual Hugging Face upload is still available if you already have a signed
directory. `ovllm publish-hf` reads `HF_TOKEN` directly and disables Hugging
Face Xet transfers by default for these small artifacts, which avoids local
token-cache and Xet-cache permission issues:

```bash
# bash/zsh
export HF_TOKEN=<your-huggingface-write-token>
export HF_HUB_DISABLE_XET=1
ovllm publish-hf <user-or-org>/gpt10m-shakespeare dist/gpt10m-shakespeare
```

```powershell
# PowerShell
$env:HF_TOKEN = "<your-huggingface-write-token>"
$env:HF_HUB_DISABLE_XET = "1"
ovllm publish-hf <user-or-org>/gpt10m-shakespeare dist/gpt10m-shakespeare
```

Build an Ollama artifact from the generated `Modelfile`:

```bash
ovllm ollama-build gpt10m-shakespeare dist/gpt10m-shakespeare
```

Dry-run wrappers are available for non-signing publish/build command-shape checks:

```bash
ovllm publish-hf <repo-id> dist/gpt10m-shakespeare --dry-run
ovllm ollama-build gpt10m-shakespeare dist/gpt10m-shakespeare --dry-run
```

Signing is intentionally performed by the GitHub Actions workflow, because the
published Sigstore identity should be the repository workflow identity.

## Reproducibility Matrix

The experiment harness tests when "same code + same seed" really means "same
model." It separates two ideas that are often conflated:

- **Run-to-run reproducibility:** train the same config twice on the same
  hardware. Do the final bits match?
- **Agreement with fp32 reference:** does a lower-precision run match fp32, or
  only produce a similar loss?

Example commands:

```bash
python run_experiment.py --model gpt10m --precision fp32 --deterministic on --device cuda
python run_experiment.py --model gpt10m --precision tf32 --deterministic on --device cuda
python run_experiment.py --model gpt10m --precision fp32 --deterministic off --device cuda --track-divergence
python sweep.py --device cuda --track-divergence
```

The matrix records:

- final loss
- tensor hash
- Merkle root
- first divergence step
- reproducible true/false
- hardware and precision metadata

## Artifact Integrity

OpenVerifiableLLM uses:

- **safetensors** for stable weight bytes.
- **tensor SHA-256** for semantic model equality.
- **raw artifact SHA-256** for file integrity.
- **Merkle chunking** for scalable partial verification.
- **Sigstore/model-transparency** for publisher identity and transparency-log
  provenance.

The Merkle manifest is computed in one read pass so file size, file hash, and
chunk hashes describe the same artifact version.

The verifier depends on the current `model-signing` CLI surface and a modern
PyYAML wheel, so those versions are pinned in both `requirements.txt` and
`pyproject.toml` for clean-machine installs.

## Threat Model

OpenVerifiableLLM catches:

- accidental corruption
- modified weights
- manifest/weight mismatch
- dataset or config drift when encoded in the manifest
- unsigned or wrongly signed published artifacts
- replay-window divergence when segment replay metadata is present

It does **not** prove that every training step was honest. A determined publisher
could construct a fraudulent but internally consistent checkpoint chain. This is
a known limitation of proof-of-learning style systems. The practical goal here is
falsifiability: make tampering and drift easy to detect, make claims reproducible,
and expose the exact assumptions under which verification holds.

## Repository Map

| Path | Purpose |
|---|---|
| `src/ovllm.py` | Verifier/publish CLI |
| `src/verifier.py` | Local/HF model verification checks |
| `src/publish.py` | Publish directory, Sigstore, HF, and Ollama helpers |
| `src/artifacts.py` | SHA-256, tensor hashing, safetensors, Merkle helpers |
| `src/experiment.py` | Shared experiment runner |
| `src/reproducibility.py` | Segmented replay and falsifiability scenarios |
| `src/signing.py` | Legacy/local Ed25519 verify-before-load helper |
| `run_experiment.py` | Run one matrix cell |
| `sweep.py` | Run the reproducibility matrix |
| `demo.py` | Narrative demo |
| `tests/` | Artifact, verifier, signing, and determinism tests |
| `RUNBOOK.md` | Demo-day commands and GPU run instructions |

## Development

Run the test suite:

```bash
python -m unittest discover -s tests
```

CUDA-only tests (TF32 divergence, determinism-OFF, DDP) self-skip on CPU
machines. Sigstore signing tests are mocked, so the full suite passes locally
without GitHub Actions OIDC credentials. The `--allow-unsigned` flag skips the
live Sigstore bundle check for local development only.

Compile-check touched modules:

```bash
python -m py_compile src/artifacts.py src/verifier.py src/publish.py src/ovllm.py
```

Check the verifier locally:

```bash
ovllm prepare-publish --weights mid_checkpoint.safetensors --out C:\tmp\ovllm-smoke
ovllm verify C:\tmp\ovllm-smoke --allow-unsigned --skip-replay
```

- Bit-exact reproducibility is scoped to a fixed hardware/software stack.
- Cross-GPU reproducibility is measured, not assumed.
- Single-GPU deterministic training is the primary supported baseline.
- Multi-GPU determinism remains experimental.
- Sigstore signing requires a real OIDC/auth environment for the deployed path.

## References

- Jia et al., *Proof-of-Learning: Definitions and Practice* (2021)
- Fang et al., *"Proof-of-Learning" Is Currently More Broken Than You Think*
  (EuroS&P 2023)
- [safetensors](https://huggingface.co/docs/safetensors)
- [Sigstore model-transparency](https://github.com/sigstore/model-transparency)

## License

See [LICENSE](LICENSE).
