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
import os
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


RECIPE_FIELDS = ["vocab_size", "max_seq_len", "embed_dim", "num_layers", "num_heads"]


def source_tensor_names(layers):
    """The checkpoint tensors the export reads, in the order `tensors` reads them."""
    names = ["transformer.wte.weight", "transformer.wpe.weight"]
    for i in range(layers):
        for src in ["ln_1", "attn.c_attn", "attn.c_proj", "ln_2", "mlp.0", "mlp.3"]:
            names += [f"transformer.h.{i}.{src}.weight", f"transformer.h.{i}.{src}.bias"]
    return names + ["transformer.ln_f.weight", "transformer.ln_f.bias"]


def same_file(a, b):
    """True if both paths name one file, also through a different spelling or a hard link."""
    if a.resolve() == b.resolve():
        return True
    try:
        return b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def read_json(path, what):
    if not path.is_file():
        sys.exit(f"{what} {path} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        sys.exit(f"{what} {path} is invalid: {exc}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="run directory with recipe.json and the *-model.pt files")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default="conversation", choices=["conversation", "base"])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run, out = Path(a.run), Path(a.out)
    # Everything is checked before anything is written.
    recipe_path = run / "recipe.json"
    cfg = read_json(recipe_path, "recipe")
    cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(cfg, dict):
        sys.exit(f"recipe {recipe_path} has no model section")
    for field in RECIPE_FIELDS:
        value = cfg.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            sys.exit(f"recipe {recipe_path}: model.{field} must be a positive integer, got {value!r}")

    weights = run / f"{a.model}-model.pt"
    if not weights.is_file():
        sys.exit(f"source checkpoint {weights} not found")
    if same_file(weights, out):
        p.error("--out must not overwrite the source checkpoint")
    try:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises many types for a damaged file
        sys.exit(f"source checkpoint {weights} is invalid: {exc}")
    if not isinstance(sd, dict):
        sys.exit(f"source checkpoint {weights} is not a state dict")
    missing = [n for n in source_tensor_names(cfg["num_layers"]) if not isinstance(sd.get(n), torch.Tensor)]
    if missing:
        shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        sys.exit(f"source checkpoint {weights} is missing tensors: {shown}")

    tok_path = Path(a.tokenizer)
    tok = read_json(tok_path, "tokenizer")
    tok = tok.get("model") if isinstance(tok, dict) else None
    if not isinstance(tok, dict) or not isinstance(tok.get("vocab"), dict) or not isinstance(tok.get("merges"), list):
        sys.exit(f"tokenizer {tok_path} must have model.vocab and model.merges")
    if not all(isinstance(t, str) and isinstance(i, int) and not isinstance(i, bool) for t, i in tok["vocab"].items()):
        sys.exit(f"tokenizer {tok_path}: model.vocab must map tokens to integer ids")

    vocab = sorted(tok["vocab"].items(), key=lambda kv: kv[1])
    if [i for _, i in vocab] != list(range(len(vocab))):
        sys.exit("tokenizer ids must be contiguous")
    if len(CONTROLS) + len(vocab) != cfg["vocab_size"]:
        sys.exit("vocabulary size differs from the recipe")
    merges = []
    for m in tok["merges"]:
        if isinstance(m, str):
            merges.append(m)
        elif isinstance(m, list) and len(m) == 2 and all(isinstance(x, str) for x in m):
            merges.append(" ".join(m))
        else:
            sys.exit(f"tokenizer {tok_path}: invalid merge {m!r}")

    # Written beside the target and renamed only once complete, so a failure halfway
    # never leaves a broken .gguf where a good one is expected.
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        write(a, cfg, sd, vocab, merges, weights, tmp)
        reader = gguf.GGUFReader(tmp)
        written = len(reader.tensors)
        del reader  # releases its memory map before the rename
        expected = 2 + 12 * cfg["num_layers"] + 3
        if written != expected:
            raise ValueError(f"wrote {written} tensors, expected {expected}")
        os.replace(tmp, out)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"could not write {out}: {exc}")
    print(f"wrote {out}")


def write(a, cfg, sd, vocab, merges, weights, path):
    w = gguf.GGUFWriter(path, "gpt2")
    # Closed even when writing fails, so the file is released and can be removed.
    try:
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

        for name, t in tensors(sd, cfg["num_layers"]):
            w.add_tensor(name, t.float().contiguous().numpy().astype(np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
    finally:
        w.close()


if __name__ == "__main__":
    main()
