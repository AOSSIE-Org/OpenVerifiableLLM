# RUNBOOK — demo day

Exact commands for the live demo. Everything except the GPU-only failure cells was
verified on CPU; the GPU cells must be run once on the pod **before** you build
slides around them (TF32 and bf16 are reliable; determinism-OFF and cross-GPU are
empirical — verify them first).

The fast path: `python demo.py --full` on the pod runs the whole arc. The phases
below are for when you want to drive each exhibit yourself.

---

## 0. Pod setup (RunPod PyTorch template, L4 / A40-class)

```bash
git clone <your repo> && cd OpenVerifiableLLM
pip install -r requirements.txt          # torch, numpy, safetensors, pynacl, matplotlib, tqdm
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
python src/signing.py                     # generates keys/ (ed25519) + self-check
```

The first text run auto-downloads tinyshakespeare (~1 MB). For the bigger corpora:
`--dataset enwik8` (downloads ~36 MB) or `--dataset wikitext` (needs `pip install datasets`).

---

## Phase B. Published-model verification loop

This is the midterm deliverable path: publish a small signed model and verify it
with one command from a clean environment.

Install the repo CLI:

```bash
pip install -r requirements.txt
pip install -e .
ovllm --help
```

Local CLI install smoke after changing `pyproject.toml` or CLI wiring:

```bash
pip install -e .
ovllm --help
python src/ovllm.py --help   # optional fallback; should show the same subcommands
```

Expected subcommands:

```text
verify
prepare-publish
sign
publish-hf
ollama-build
```

Prepare a publish directory from the real trained safetensors artifact:

```bash
ovllm prepare-publish \
  --weights artifacts/gpt10m_shakespeare_fp32_deton_s99.safetensors \
  --out dist/gpt10m-shakespeare \
  --name gpt10m-shakespeare
```

The generated model card should say the repo includes:

- safetensors weights
- `ovllm_manifest.json`
- Sigstore/model-transparency bundle
- Merkle metadata

Rerun `prepare-publish` after model-card or manifest template changes so the
publish directory contains the current generated metadata.

Do not locally sign the published artifact. The project-scoped signing path is
the **Publish Verified Model** GitHub Actions workflow, which uses GitHub OIDC
so the Sigstore identity points at the workflow, not a personal local browser
session.

Local signing is disabled by default to prevent developers from accidentally signing
with their personal accounts. If you need to test signing locally, set the environment
variable `OVLLM_ALLOW_LOCAL_SIGNING=true` (or `$env:OVLLM_ALLOW_LOCAL_SIGNING="true"` in PowerShell).

Run the workflow with:

```text
model_name: gpt10m-shakespeare
hf_repo_id: <user-or-org>/gpt10m-shakespeare
publish_to_hf: true
```

The workflow signs with:

```text
identity: https://github.com/<owner>/<repo>/.github/workflows/publish-verified-model.yml@<git-ref>
provider: https://token.actions.githubusercontent.com
```

It then verifies the signed directory without `--allow-unsigned`:

```bash
ovllm verify dist/gpt10m-shakespeare --skip-replay
```

If you already have a signed directory, manual Hugging Face upload uses
`HF_TOKEN` directly and disables Hugging Face Xet transfers by default for these
small artifacts. This avoids local Hugging Face CLI token-cache and Xet-cache
permission issues:

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

Verify by model reference:

```bash
# Verify metadata and signatures only (fast)
ovllm verify <user-or-org>/gpt10m-shakespeare --skip-replay

# Verify with full local training/replay verification (bit-for-bit parameter match)
ovllm verify <user-or-org>/gpt10m-shakespeare
```

Remote verification downloads into `.ovllm-cache/huggingface` by default. If a
machine has cache permission issues or you want a disposable cache, use:

```bash
ovllm verify <user-or-org>/gpt10m-shakespeare --skip-replay --cache-dir C:\tmp\ovllm-hf-cache
```

If hashes pass but `sigstore_bundle` fails with `manifest lacks
sigstore_identity/provider`, the uploaded repo was not the GitHub Actions-signed
artifact. Re-run **Publish Verified Model** with `publish_to_hf: true` and verify
the newly uploaded output.

Clean-machine smoke:

```bash
git clone <your repo> && cd OpenVerifiableLLM
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
ovllm verify <user-or-org>/gpt10m-shakespeare --skip-replay
```

Optional Ollama build path:

```bash
ovllm ollama-build gpt10m-shakespeare dist/gpt10m-shakespeare
```

Dry-run wrappers are for non-signing publish/build command-shape checks:

```bash
ovllm publish-hf <repo-id> dist/gpt10m-shakespeare --dry-run
ovllm ollama-build gpt10m-shakespeare dist/gpt10m-shakespeare --dry-run
```

Do not use local signing dry runs as part of the demo path. Published artifacts
are signed by the GitHub Actions workflow so Sigstore records the workflow
identity.

---

## 1. Headline: the same GPU is bitwise reproducible (and the audit passes)

```bash
cd src
python gpu_reproducibility_test.py        # fresh-vs-fresh: identical bits run-to-run
python reproducibility.py                 # CLEAN AUDIT PASS + the 5 scenarios
cd ..
```

What to say: "With a pinned cuBLAS workspace and deterministic algorithms, training
this 11M-param model twice on this GPU gives **the same bits** — and the segmented
replay audit confirms a checkpoint can be resumed bit-for-bit." Scenario 5 shows a
tampered checkpoint **rejected before deserialization** (the security fix).

CPU smoke (proves plumbing without waiting on a GPU):
```bash
cd src && OVL_TOTAL_STEPS=10 OVL_CHECKPOINT_STEP=5 OVL_BATCH_SIZE=4 OVL_BLOCK_SIZE=64 \
    python reproducibility.py ; cd ..
```

---

## 2. The matrix (the spread of PASS/FAIL)

```bash
python sweep.py --device cuda --track-divergence
# add the scale + modality rows:
python sweep.py --device cuda --stretch --track-divergence
```

Writes `results/results.jsonl` and prints the grid. Expect: fp32+det-on reproducible;
bf16 + tf32 bit-differ from the fp32 reference; determinism-OFF breaks run-to-run on
GPU (first-divergence step populated).

---

## 3. Single cells for slides

```bash
# control
python run_experiment.py --model gpt10m --precision fp32 --deterministic on  --device cuda
# the silent killer (Ampere default): loss barely moves, bits change
python run_experiment.py --model gpt10m --precision tf32 --deterministic on  --device cuda
# half precision
python run_experiment.py --model gpt10m --precision bf16 --deterministic on  --device cuda
# determinism OFF: run-to-run divergence
python run_experiment.py --model gpt10m --precision fp32 --deterministic off --device cuda --track-divergence
```

---

## 4. Cross-GPU column (verify before relying on it)

`use_deterministic_algorithms` holds **per hardware stack**, not across architectures.

```bash
# On GPU type A (e.g. L4):
python sweep.py --device cuda --out results/results_L4.jsonl
# Stop the pod, switch the GPU type to A40, restart, then:
python sweep.py --device cuda --out results/results_A40.jsonl
# Compare param hashes per cell:
python - <<'PY'
import json
load=lambda p:{(o["model"],o.get("condition")):o["param_sha256"] for o in map(json.loads,open(p))}
a,b=load("results/results_L4.jsonl"),load("results/results_A40.jsonl")
for k in sorted(a):
    print(f"{k[0]:>7} {str(k[1]):<12} {'SAME' if a[k]==b.get(k) else 'DIFF'}")
PY
```

Then fill the demo's CROSS-GPU column:
```bash
python demo.py --full --device cuda --cross-gpu-results results/results_A40.jsonl
```

---

## 5. T7 — the divergence-accumulation plot

```bash
# determinism OFF, hashing params every step in two twin runs:
python run_experiment.py --model gpt10m --precision fp32 --deterministic off \
    --device cuda --track-divergence --out results/divergence.jsonl
python src/plot_divergence.py results/divergence.jsonl   # -> results/divergence.png
```

The plot marks the step where two "identical" runs first differ — the most novel
artifact in the deck.

---

## 6. Stretch (only if green)

```bash
python run_experiment.py --model gpt50m --precision fp32 --deterministic on --device cuda   # scale axis
python run_experiment.py --model cnn --dataset cifar --precision fp32 --deterministic off --device cuda  # conv nondeterminism
# DDP (>=2 GPUs): all-reduce ordering vs bitwise repro
torchrun --nproc_per_node=2 src/ddp_repro.py
```

A negative DDP result (cannot get bitwise-identical across ranks) is still the real
open problem — present it as such.

---

## 7. Dry run + fallback

```bash
time python demo.py --full --device cuda          # one timed full pass the night before
```

Record a clean `demo.py --full` screen-capture as a fallback in case the pod is flaky
on stage. Keep `results/results.jsonl` from a good run as a static backup for the grid.

---

## 8. The k-of-N chain audit (mid-scale evidence run)

The mechanism behind the README's "What a spot-check audit buys": a prover seals a
signed full-state boundary checkpoint every S steps; an auditor samples k of the N
segments, replays each bit-exactly, and the report measures the realized audit-cost
ratio instead of asserting it.

CPU smoke (~a minute, proves the plumbing):

```bash
cd src
python chain.py train --model mlp --segments 4 --segment-steps 3 --batch-size 4 --block-size 48
python chain.py audit --k 2 --audit-seed 7          # exit 0 iff every sampled segment verifies
cd ..
```

Pod scale — the run the write-up hangs on (gpt120m ≈ 116M params, enwik8):

```bash
cd src
python chain.py train --model gpt120m --dataset enwik8 --segments 20 --segment-steps 500 --device cuda
python chain.py audit --k 3 --audit-seed 7 --device cuda
cd ..
```

Numbers to record from the audit report: `cost_ratio` (auditor wall time / prover
wall time — should approach k/N as segment compute grows), per-segment wall times,
chain storage (`du -sh runs/chain`), and `min_forgery_detection_probability`.

**Known caveat (measured, not hidden):** at smoke scale the cost ratio EXCEEDS k/N
(fixed per-segment overhead — model build, dataset load, checkpoint I/O — dominates
a 3-step segment). The k/N economics only hold when segment compute dominates that
overhead; the mid-scale run is what demonstrates the crossover.

**Measured (2026-07-07, RunPod secure A40, torch 2.11.0+cu128 — evidence in
`proofs/chain_a40/`):** gpt120m (116,343,808 params) on enwik8, 20 × 500 steps,
strict deterministic fp32. All 20 segments sealed at a steady ~63.6 s; the k=3
audit (segments 4, 10, 12, audit_seed=7) replayed **bit-exactly** — GPU bitwise
determinism holds at 116M params through full save/restore boundaries. Auditor
274.5 s vs prover 1589.0 s → **measured cost ratio 0.173 vs theoretical k/N =
0.15** (replay overhead ≈ 1.44× a prover segment: checkpoint load + hash vs
checkpoint save). Chain storage: 28.5 GB for 21 boundaries ≈ 1.36 GB/boundary
(fp32 weights 465 MB + Adam moments 2×). Total pod cost ≈ $0.32.

**Cross-hardware controls (2026-07-07, evidence in `proofs/chain_cross/`):** the
same gpt120m chain (5 × 200 steps, enwik8, trained on a secure RTX A4000) was
audited with identical sampled segments (1 and 2, audit_seed=7) from three
positions: the training pod itself (**PASS**, ratio 0.460), a second RTX A4000
pod on a different physical GPU (**PASS**, bit-exact, ratio 0.461), and an L4 pod
in a different datacenter (**FAIL**, closing hash mismatch on both segments;
opening hashes matched, so the divergence is replay arithmetic, not transfer
corruption). Auditors pulled the 7.5 GB chain over plain HTTP; ed25519 signatures
established integrity. Conclusion: the verification equivalence class is (GPU
model, software stack) — an auditor needs the same GPU model, not the same
machine. Three-pod total cost ≈ $0.20.

Tamper drill (any byte edit to a boundary file fails its segment before
deserialization; a manifest edit fails the audit immediately):

```bash
printf '\xff' | dd of=runs/chain/boundary_0001.pt bs=1 seek=1024 conv=notrunc
python src/chain.py audit --segments 1   # -> FAIL (signature), exit 1
```

---

## 9. Replay-free forgery detectors (segment-length sweep)

`src/forgery.py` trains a genuine chain plus an alt-seed donor, plants forgeries
(splice, gradual interpolation, gaussian edits), and scores three O(params)
replay-free detectors against the genuine chain's envelope. CPU smoke:

```bash
cd src && python forgery.py --model mlp --segments 4 --segment-steps 3 \
    --batch-size 4 --block-size 48
```

Pod scale (use a fast Ada card — L40S ran gpt120m at ~14 s per 200-step segment):

```bash
OVL_FORGE_SEGMENTS=10 OVL_FORGE_SEGMENT_STEPS=100 bash scripts/pod_forgery_run.sh
```

**Measured (2026-07-07, L40S, gpt120m/enwik8, evidence in `proofs/forgery_l40s/`):**
zero false positives in every condition; alt-seed splice and α=0.10 interpolation
detected+localized at all segment lengths; the stealthy α=0.03 interpolation is
missed at S=200 but detected+localized at S=100 and S=10 — segment length is a
replay-free detectability dial. σ=1e-2 gaussian edits evade the detectors at this
scale (caught only by sampled replay / the final-model hash). Genuine
moment-cosine range rises from [0.013, 0.050] at S=200 to [0.052, 0.128] at S=10,
confirming the correlation-decay mechanism.

**Smart-forger escalation (same day, second L40S sweep, evidence in
`proofs/smartforger_l40s/`):** all three moment-forging attackers detected AND
localized at every segment length, zero false positives. sf-aligned (cos=1
moments) caught by the two-sided moment envelope everywhere; sf-calibrated
(moments engineered into the genuine envelope, genuine v copied) evades the
moment checks at S=200/100 but is caught by weight-side reach/norm — and at S=10
the moment envelope flags it as well; sf-freshv (fabricated v) trips the
elementwise hard invariant v_{k+1} >= beta2^S * v_k at every S. Net: moment
forging pays only when the weight delta is also small; that small-delta/long-S
hole is what shorter segments and sampled replay close.

---

## Troubleshooting

- **"deterministic algorithm not available"** on an exotic op: the sweep already runs
  with `warn_only=True`; the audit's control uses strict mode. If the audit crashes on
  a specific op, run that cell through `sweep.py` (warn_only) instead.
- **CUBLAS_WORKSPACE_CONFIG**: set automatically on import of `device`. If you see a
  cuBLAS determinism error, confirm nothing imported torch and touched CUDA before
  `device`/`main`.
- **tf32 shows SAME on CPU**: expected — TF32 is an Ampere+ GPU feature; the divergence
  only appears on `--device cuda`.
- **Dataset download blocked**: `--dataset shakespeare` falls back to the bundled
  `data/shakespeare_sample.txt`; determinism results stay valid.
