# Run the model in Ollama

The released chat model can run in [Ollama](https://ollama.com) and other llama.cpp-based apps. It
is exported to GGUF with the model's own tokenizer and chat format.

This is for trying the model. It isn't how the model is verified. llama.cpp uses a slightly
different GELU and different kernels, so its outputs are close to PyTorch's but not bit-identical.
To check the model, use the released `.pt` files and [`ovl.py verify`](check_a_segment.md).

## Run it

The model is on the Ollama registry as
[`ryoari/openverifiablellm`](https://ollama.com/ryoari/openverifiablellm):

```bash
ollama run ryoari/openverifiablellm "Tell me about the Moon."
```

The GGUF file is also attached to the
[v1.0.0 release](https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/tag/v1.0.0) and stored in the
[Hugging Face dataset](https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/tree/2fbebaa3ac861898eb8a3dbd9a2a44d08d08216c/release/simple-run/gguf)
(revision `2fbebaa3`). All three are the same file, SHA-256
`541d49c97b87cd4ed0acb90b4327535ee20b8c11c452e2583d460154e9d980de`. Running
`ollama show --modelfile ryoari/openverifiablellm` prints a `FROM` line ending in that hash.

## Build it yourself

You need the release's run folder and tokenizer (see [Check a training segment](check_a_segment.md)
for the download), plus `pip install gguf torch`.

```bash
python simple/export_gguf.py --run RUN --tokenizer PREP/tokenizer/tokenizer.json --out ovl-23m-chat.gguf
ollama create openverifiablellm -f simple/Modelfile     # run where ovl-23m-chat.gguf is
ollama run openverifiablellm "Tell me about the Moon."
```

`--model base` exports the Wikipedia-only model instead, which continues text rather than chatting.
The GGUF's description records the SHA-256 of the `.pt` file it came from.

## What the export contains

| | |
|---|---|
| Architecture | llama.cpp `gpt2`: 6 layers, 6 heads, 384 dimensions, 512-token context |
| Weights | float32, unchanged from the released `.pt` file (143 MB) |
| Tokenizer | GPT-2-style byte-level BPE, 32,000 tokens: `<\|pad\|>`, `<\|bos\|>`, `<\|eos\|>`, `<\|user\|>` and `<\|assistant\|>` at ids 0–4, then the trained vocabulary |
| Chat format | `<\|user\|>` question `<\|eos\|>` `<\|assistant\|>` answer `<\|eos\|>`, as in training |

Running the exported tensors with llama.cpp's GPT-2 computation reproduces the PyTorch model's
greedy output token for token on a test prompt.
