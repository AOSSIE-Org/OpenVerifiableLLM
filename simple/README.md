# `simple/`: the verifiable training loop

Everything that trained, checked and published the released model. `ovl.py` is the core: one file
that trains with a hash-linked record of every checkpoint, verifies any segment bit for bit, forks
a run to remove documents, and chats with the result. The shell scripts wrap it for rented GPU
pods. For the method and results, see [how it works](../docs/how-it-works.md); to check the release,
see [verify](../docs/verify.md).

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
| Run the model in Ollama | `ollama run ryoari/openverifiablellm`, or build it yourself ([below](#run-the-model-in-ollama)) | any CPU |

`RUN` is a run directory (`chain.jsonl`, `recipe.json`, models, `ckpt-*.pt`). `PREP` is the prepared
data directory, `prepared/8de1d600…` in the
[release dataset](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence).
On a CPU, set `OVL_DEVICE=cpu`.

## Files

| File | What it does |
|---|---|
| [`ovl.py`](ovl.py) | The loop. Commands: `train`, `verify` (`--weights`, `--segment`, `--random`, `--all`, `--data`; it needs the run's `anchor.json` and its two timestamps, or `--anchor FILE`; `--no-anchor` and `--no-timestamps` are only for a run that hasn't been anchored yet), `fork`, `chat`, `bench` |
| [`model.py`](model.py), [`data.py`](data.py) | The GPT model, and the token ids, batch layout and seeding shared with the data preparation (copied unchanged from commit `3f882ad`) |
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
| [`export_safetensors.py`](export_safetensors.py) | Exports a released model to safetensors with every weight unchanged; `verify --weights` accepts the result |
| [`export_gguf.py`](export_gguf.py), [`Modelfile`](Modelfile) | Export a released model to GGUF for llama.cpp and Ollama, and the Ollama build file with the training chat format |

## Requirements

- Python 3.12, PyTorch and NumPy. `chat` and `loo_eval.py` also need `tokenizers==0.23.2`, and
  `export_gguf.py` needs `gguf`. The shell scripts install what they need. For development, use the
  hash-locked environment in [CONTRIBUTING.md](../CONTRIBUTING.md#development-setup).
- Bit-exact training and segment checks: an RTX 5090 with the image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8). Other GPU
  architectures don't reproduce the bits; see [hardware](../docs/how-it-works.md#results).

Tests: `OVL_DEVICE=cpu OVL_TIMESTAMP=0 python -m pytest tests`.

## Run the model in Ollama

The chat model is on the Ollama registry as
[`ryoari/openverifiablellm`](https://ollama.com/ryoari/openverifiablellm). The GGUF file is also
attached to each [release](https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases). All copies are
the same file, SHA-256 `541d49c97b87cd4ed0acb90b4327535ee20b8c11c452e2583d460154e9d980de`;
`ollama show --modelfile ryoari/openverifiablellm` prints a `FROM` line that ends in that hash.

This is for trying the model, not for verifying it. llama.cpp uses a slightly different GELU and
different kernels, so its outputs are close to PyTorch's but not bit-identical. To check the model,
use the released `.pt` files and `ovl.py verify`.

To build it yourself, you need the run folder and the tokenizer, and `pip install gguf torch`:

```bash
python simple/export_gguf.py --run RUN --tokenizer PREP/tokenizer/tokenizer.json --out ovl-23m-chat.gguf
cp simple/Modelfile .                                     # its FROM line is ./ovl-23m-chat.gguf
ollama create openverifiablellm -f Modelfile
ollama run openverifiablellm "Tell me about the Moon."
```

`--model base` exports the Wikipedia-only model instead; it continues text and doesn't chat. The
GGUF's description records the SHA-256 of the `.pt` file it came from.

| | |
|---|---|
| Architecture | llama.cpp `gpt2`: 6 layers, 6 heads, 384 dimensions, 512-token context |
| Weights | float32, unchanged from the released `.pt` file (143 MB) |
| Tokenizer | GPT-2-style byte-level BPE, 32,000 tokens: `<\|pad\|>`, `<\|bos\|>`, `<\|eos\|>`, `<\|user\|>` and `<\|assistant\|>` at ids 0–4, then the trained vocabulary |
| Chat format | `<\|user\|>` question `<\|eos\|>` `<\|assistant\|>` answer `<\|eos\|>`, as in training |

Running the exported tensors with llama.cpp's GPT-2 computation reproduces the PyTorch model's
greedy output token for token on a test prompt.
