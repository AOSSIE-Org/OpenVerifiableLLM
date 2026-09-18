# GPU development pilot and complete update census

This path is development infrastructure. A complete actual CUDA synthetic cycle
passed: eight recorded updates, full fresh-process replay from regenerated
initialization, and a separate four-update resume. All nine saved states also
passed checks after a fresh public download. See
[the fifth live attempt](../project/evidence/live-tiny-cuda-v5/README.md).
Sustained representative throughput and production admission remain pending.
Operator-run replay is not independent third-party verification.

## Numerical profile

`ovl_pipeline.gpu` requires exactly one visible CUDA device and one physical GPU in
`nvidia-smi`. PyTorch must be an explicitly pinned 2.14.0 CUDA build. It refuses a
CPU fallback, an already initialized CUDA process, unsupported precisions and TF32
overrides. Start each record/replay in a fresh process with:

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLASLT_WORKSPACE_SIZE=32768 TORCH_CUBLASLT_UNIFIED_WORKSPACE=1
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0 USE_PYTORCH_KERNEL_CACHE=0
```

The closed kernel object is `{"schema":"ovl.gpu-kernel.v1","precision":"bf16"}`
or the explicit `fp32` alternative. CPU initialization is regenerated from the
recipe; FP32 master parameters are then moved to CUDA. The existing manual-attention
model and shared update kernel are used, with explicit nonfused/noncapturable AdamW,
no gradient scaler, no compilation and BF16 autocast cache disabled. Coverage and
batch transcripts bind the original CPU batch tensors. Finite loss, gradients and
parameters are checked at every update.

The profile uses PyTorch 2.14's IEEE precision API, deterministic algorithms with
errors enabled, deterministic uninitialized-memory filling and disabled reduced
precision/split-K reductions. See the official [CUDA precision
notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html),
[reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html) and
[autocast operation reference](https://docs.pytorch.org/docs/2.14/amp.html).
The setters and every update assert the literal required flag profile; merely
recording a self-consistent value is insufficient. Configuration flags alone do not guarantee reproducibility across hardware or
software versions.

The backend preference is explicitly cuBLASLt and is read back before every
update. This is a preference setting, not proof of exclusive kernel dispatch.
Both BLAS workspace sizes are bound to 32 MiB; the Lt environment variable uses
KiB. Actual getters are checked after explicit CUDA initialization and before
updates, and recorded in the compatible environment. Reconfiguration rejects
existing flag/workspace drift before setting anything. The audited child receives
a fixed executable path and locale, a fresh evidence HOME, deterministic settings
and only the selected activity path; operator credentials and arbitrary shell
overrides are not inherited.

The third real tiny attempt exposed the missing backend preference and failed
before recording training. PyTorch's BF16 cuBLAS GEMM guard uses an FP16-labelled
error string in the [pinned upstream source](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/cuda/CUDABlas.cpp#L1074).
Host-only setters do not trigger that GEMM guard. CPU tests and real CUDA-wheel
host readback checks therefore remain configuration evidence; the next actual
GPU record/replay is required to validate the correction numerically.

After warmup, the environment report records driver/build/device properties,
precision flags, relevant environment variables, numerical package RECORD hashes
and hashes of loaded numerical libraries. Physical GPU UUID is reported separately
from the compatible-environment identity. A future production registration must also
pin an immutable container and complete wheel lock; this report does not attest a
host or replace that requirement.

## Record, continuous replay and separate resume probe

Provisioning, billing supervision and provider deadlines are outside this CLI.
They must be installed and verified before any paid pilot. Each output directory
must be fresh; preserve failed attempts. Use representative prepared Wikipedia and
conversation streams separately. Pilot cycling repeats the stream for development
measurement only. Never use pilot weights as production initialization.

```sh
python -m ovl_pipeline.gpu_pilot record \
  --recipe recipe.json --kernel kernel.json --stream prepared/wikipedia \
  --seconds 600 --warmup-updates 4 --checkpoint-every 128 --output pilot/wiki
python -m ovl_pipeline.gpu_pilot replay \
  --stream prepared/wikipedia --record-directory pilot/wiki \
  --expected-record-sha256 CALLER_SELECTED_RECORD_SHA256 --output replay/wiki
python -m ovl_pipeline.gpu_pilot replay \
  --stream prepared/wikipedia --record-directory pilot/wiki \
  --expected-record-sha256 CALLER_SELECTED_RECORD_SHA256 \
  --resume-from 1 --output resume-probe/wiki
```

The CLI prints the canonical record digest. Select it from the retained record
receipt, not from an untrusted replacement. Short fixed-update probes use
`--updates N` instead of `--seconds`; they are not eligible throughput measurements.
Choose a noninitial, nonfinal checkpoint index for the resume probe.

Warmup weights are discarded and all initial numerical/RNG/control state is
regenerated. Timed measurements exclude warmup and initial checkpoint creation,
include subsequent logging/checkpoint costs, and synchronize CUDA before stopping
the timer. Setup time includes full stream validation, warmup, library fingerprinting
and the initial checkpoint. Account for setup separately in actual provider spend.
The final short batch and any cycling overhead remain inside elapsed time.

Continuous replay regenerates the initial state and compares every scheduled safe
checkpoint without restoring prover checkpoints. The separately labeled resume probe
restores a recorded intermediate checkpoint and recomputes the remaining suffix.
Both are performed by the project operator and are not independent verification.
All reports leave production coverage and admission `NOT_RUN`.

## Full-corpus census and cost arithmetic

```sh
python -m ovl_pipeline.coverage --stream prepared/wikipedia \
  --recipe recipe.json --output wikipedia-schedule.json
python -m ovl_pipeline.coverage --stream prepared/conversation \
  --recipe recipe.json --output conversation-schedule.json
```

The census hashes and validates every stream file and document, then independently
counts all target-bearing windows and updates without invoking the training batch
generator. It binds the stream and recipe digests and includes final short batches.
It proves a planned count, not completed training coverage.

Cost input `ovl.cost-forecast-input.v3` uses complete `updates`, completed training
and replay update counts, `measured_full_batch_updates`, `measured_ms`, and the
measurement/replay/recipe/stream/schedule SHA-256 digests for both phases. It also
requires timed-pilot eligibility, all measured updates, checkpoint counts and
intervals, full production checkpoint counts (including recovery saves), and full
continuous replay timing. Pilot checkpoint density must cover the complete planned
production density; setup and boundary-zero costs are reserved separately. Both
remaining paths use the slower of recording and replay rates. Resume probes and
fixed-update probes cannot supply this measurement. A pilot must last
at least ten minutes after warmup with representative checkpoint overhead. Its full
batch count is the rate denominator; all elapsed work, including short batches,
remains in the numerator. Charge every production update at this rate, including
its last short batch, and retain the 25% runtime margin.

Useful-target throughput is also reported, but cannot alone price arbitrary
short-document padding. An adversarial census test holds targets constant while
increasing required updates by 16x. Historical v1/v2 arithmetic remains readable but
must not admit production. No arithmetic version authenticates its input evidence or
implements a live provider guard. The production controller must bind the actual
complete census, compatible kernel and real pilot record/replay, reconcile spend,
reserve setup/reconstruction/export/storage costs, and enforce the $90 operating
limit with $10 protected reserve.
