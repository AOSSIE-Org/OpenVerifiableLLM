"""Export a released model to GGUF, the file format llama.cpp and Ollama load.

    python simple/export_gguf.py --run RUN --tokenizer PREP/tokenizer/tokenizer.json --out ovl-23m-chat-F32.gguf
    python simple/export_gguf.py ... --model base      # the Wikipedia-only model instead of the chat model

The model is a GPT-2-style transformer and the tokenizer a GPT-2-style byte-level BPE, so the file
uses llama.cpp's `gpt2` architecture and tokenizer. The five control tokens (pad, bos, eos, user,
assistant) take ids 0-4, exactly as in training, and the BPE vocabulary follows at id 5.
Weights are stored as float32, unchanged. Needs `pip install gguf`.

The export is for convenient use, not for verification: llama.cpp's GELU is the tanh
approximation and its kernels differ, so outputs are close to PyTorch's but not bit-identical.
Check the released .pt files with `ovl.py verify`.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import gguf
import numpy as np
import torch

CONTROLS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]  # ids 0-4, see data.py
# Single and multi-turn chat in the training format: <|user|> text <|eos|> <|assistant|> text <|eos|> ...
CHAT_TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'user' %}<|user|>{{ m['content'] }}<|eos|>"
                 "{% elif m['role'] == 'assistant' %}<|assistant|>{{ m['content'] }}<|eos|>{% endif %}{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def tensors(sd, layers):
    yield "token_embd.weight", sd["transformer.wte.weight"]
    yield "position_embd.weight", sd["transformer.wpe.weight"]
    for i in range(layers):
        p, b = f"transformer.h.{i}.", f"blk.{i}."
        for src, dst in [("ln_1", "attn_norm"), ("attn.c_attn", "attn_qkv"), ("attn.c_proj", "attn_output"),
                         ("ln_2", "ffn_norm"), ("mlp.0", "ffn_up"), ("mlp.3", "ffn_down")]:
            yield b + dst + ".weight", sd[p + src + ".weight"]
            yield b + dst + ".bias", sd[p + src + ".bias"]
    yield "output_norm.weight", sd["transformer.ln_f.weight"]
    yield "output_norm.bias", sd["transformer.ln_f.bias"]
    yield "output.weight", sd["transformer.wte.weight"]  # tied with the token embedding


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="run directory with recipe.json and the *-model.pt files")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default="conversation", choices=["conversation", "base"])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run = Path(a.run)
    recipe_file = run / "recipe.json"
    if not recipe_file.is_file():
        sys.exit(f"recipe file {recipe_file} not found")
    tok_file = Path(a.tokenizer)
    if not tok_file.is_file():
        sys.exit(f"tokenizer file {tok_file} not found")
    weights = run / f"{a.model}-model.pt"
    if not weights.is_file():
        sys.exit(f"source checkpoint {weights} not found")
    if weights.resolve() == Path(a.out).resolve() or (Path(a.out).exists() and weights.samefile(a.out)):
        p.error("--out must not overwrite the source checkpoint")
    try:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict) or "transformer.wte.weight" not in sd:
            sys.exit(f"source checkpoint {weights} is invalid")
    except Exception as exc:
        sys.exit(f"source checkpoint {weights} is invalid: {exc}")
    cfg = json.loads(recipe_file.read_text(encoding="utf-8"))["model"]
    tok = json.loads(tok_file.read_text(encoding="utf-8"))["model"]
    vocab = sorted(tok["vocab"].items(), key=lambda kv: kv[1])
    if [i for _, i in vocab] != list(range(len(vocab))):
        sys.exit("tokenizer ids must be contiguous")
    if len(CONTROLS) + len(vocab) != cfg["vocab_size"]:
        sys.exit("vocabulary size differs from the recipe")
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["merges"]]
    try:
        model_tensors = list(tensors(sd, cfg["num_layers"]))
    except (KeyError, TypeError) as exc:
        sys.exit(f"source checkpoint {weights} is missing required tensor: {exc}")

    w = gguf.GGUFWriter(a.out, "gpt2")
    w.add_name(f"OpenVerifiableLLM 23M {a.model}")
    w.add_description(f"Exported from {weights.name} (sha256 {hashlib.sha256(weights.read_bytes()).hexdigest()})")
    w.add_context_length(cfg["max_seq_len"])
    w.add_embedding_length(cfg["embed_dim"])
    w.add_feed_forward_length(4 * cfg["embed_dim"])
    w.add_block_count(cfg["num_layers"])
    w.add_head_count(cfg["num_heads"])
    w.add_layer_norm_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(CONTROLS + [t for t, _ in vocab])
    w.add_token_types([gguf.TokenType.CONTROL] * len(CONTROLS) + [gguf.TokenType.NORMAL] * len(vocab))
    w.add_token_merges(merges)
    w.add_pad_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(a.model == "base")
    if a.model == "conversation":
        w.add_chat_template(CHAT_TEMPLATE)

    for name, t in model_tensors:
        w.add_tensor(name, t.float().contiguous().numpy().astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
