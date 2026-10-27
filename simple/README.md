# `simple/`: the verifiable training loop

Everything that trained, checked and published the released model. `ovl.py` is the core: one file
that trains with a hash-linked record of every checkpoint, verifies any segment bit for bit, forks
a run to remove documents, and chats with the result. The shell scripts wrap it for rented GPU
pods. For the method and results, see [the simple run](../docs/design/simple_run.md).

## Which script do I want?

| I want to… | Run | Needs |
|---|---|---|
| Check the released weights and record | `python simple/ovl.py verify --run RUN --weights RUN/conversation-model.pt` | any CPU |
| Check one training segment of the release | `bash simple/verify_segment.sh` | RTX 5090 |
| Do that in one click on a fresh pod | the RunPod template in [`checker_template.md`](checker_template.md), which runs [`checker_pod.sh`](checker_pod.sh) | RTX 5090 pod |
| Chat with a trained model | `python simple/ovl.py chat --run RUN --tokenizer PREP/tokenizer/tokenizer.json "..."` | any CPU |
| Check the release's timestamps | `python simple/stamp.py RUN --check` | `openssl` |
| Train a model from scratch | `bash simple/run_full.sh` | RTX 5090 pod |
| Rebuild the training data from the raw dump | `bash simple/rebuild_prep.sh` | CPU, ~150 GB disk, ~1 day |
| Test whether another GPU reproduces the bits | `bash simple/hwmap_check.sh` | any GPU |
| Remove documents from a run and retrain | `python simple/ovl.py fork ...`, then `loo_eval.py` | RTX 5090 |
| Estimate the cost of a bigger model | `python simple/bench_sizes.py --data PREP` | RTX 5090 |
| Run the model in Ollama | `python simple/export_gguf.py ...`, then `ollama create -f simple/Modelfile` ([guide](../docs/guides/ollama.md)) | any CPU |

`RUN` is a run directory (`chain.jsonl`, `recipe.json`, models, `ckpt-*.pt`). `PREP` is the prepared
data directory, `prepared/8de1d600…` in the
[release dataset](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence).
On a CPU, set `OVL_DEVICE=cpu`.

## Files

| File | What it does |
|---|---|
| [`ovl.py`](ovl.py) | The loop. Commands: `train`, `verify` (`--weights`, `--segment`, `--random`, `--all`; it needs the run's `anchor.json` or `--anchor FILE`, and `--no-anchor` is only for a run that hasn't been anchored yet), `fork`, `chat`, `bench` |
| [`stamp.py`](stamp.py) | RFC 3161 timestamps (FreeTSA and DigiCert) for each chain entry while training runs; `--check` verifies them. `OVL_TIMESTAMP=0` turns stamping off |
| [`verify_segment.sh`](verify_segment.sh) | Downloads the release's data, record and the needed checkpoints, then checks `K` random segments (or `SEGMENTS="7 8"`) |
| [`checker_pod.sh`](checker_pod.sh) | One-click check for a fresh RTX 5090 pod: fetches the release code at a pinned revision, runs `verify_segment.sh`, prints one result line to share |
| [`checker_template.md`](checker_template.md) | The description and start command of the public RunPod template |
| [`run_full.sh`](run_full.sh) | Whole production loop on one pod: bench, train, spot-check segments, chat sample. Re-running resumes |
| [`follow.sh`](follow.sh) | Runs on a second pod during training: mirrors every checkpoint off the training pod and checks each segment as it closes |
| [`fetch.sh`](fetch.sh) | Parallel ranged downloads from the release dataset; sourced by the other scripts |
| [`rebuild_prep.sh`](rebuild_prep.sh) | Rebuilds the prepared token streams from the raw Wikipedia dump and compares them with the published `preparation.json` |
| [`hwmap_check.sh`](hwmap_check.sh) | Recomputes one short published segment (32→33) on whatever GPU it runs on and reports MATCH or MISMATCH |
| [`loo_eval.py`](loo_eval.py) | Compares an original run with a fork that removed documents: loss on the removed text against control articles |
| [`bench_sizes.py`](bench_sizes.py) | Times the deterministic training step for several model sizes and prints the cost of training plus a full check |
| [`review_ckpt.py`](review_ckpt.py) | Inspects checkpoints before publication: exact hashes, zip members, restricted load, allowed strings |
| [`export_gguf.py`](export_gguf.py), [`Modelfile`](Modelfile) | Export a released model to GGUF for llama.cpp and Ollama, and the Ollama build file with the training chat format |

## Requirements

- Python 3.10+, PyTorch, and the release's pinned packages:
  `pip install numpy rfc8785==0.1.4 tokenizers==0.23.2 mwparserfromhell==0.7.2 "safetensors>=0.4.5,<1.0" "pynacl>=1.5,<2.0"`
  (the shell scripts install these themselves).
- Bit-exact training and segment checks: an RTX 5090 with the image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8). Other GPU
  architectures don't reproduce the bits; see [hardware scope](../docs/design/simple_run.md#results).

Tests: `OVL_DEVICE=cpu OVL_TIMESTAMP=0 python -m pytest tests/simple`.
