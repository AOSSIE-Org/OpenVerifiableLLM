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
