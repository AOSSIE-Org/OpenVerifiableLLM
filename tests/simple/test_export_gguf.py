"""export_gguf.py writes a llama.cpp `gpt2` file whose tensors and vocabulary match the source."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]
from model import TinyGPT  # noqa: E402

CFG = {"attn_impl": "manual", "dropout": 0, "embed_dim": 16, "max_seq_len": 8, "num_heads": 2,
       "num_layers": 2, "vocab_size": 5 + 4}


def make_run(tmp_path):
    torch.manual_seed(0)
    model = TinyGPT(**CFG)
    model.lm_head.weight = model.transformer["wte"].weight
    torch.save(model.state_dict(), tmp_path / "conversation-model.pt")
    (tmp_path / "recipe.json").write_text(json.dumps({"model": CFG}), encoding="utf-8")
    tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġa": 3}, "merges": [["a", "b"]]}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    return model.state_dict()


def export(tmp_path):
    out = tmp_path / "m.gguf"
    subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                    "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(out)], check=True)
    return gguf.GGUFReader(out)


def test_tensors_and_vocabulary_match(tmp_path):
    sd = make_run(tmp_path)
    r = export(tmp_path)
    t = {x.name: np.array(x.data).reshape([int(d) for d in reversed(x.shape)]) for x in r.tensors}
    assert len(t) == 2 + 12 * CFG["num_layers"] + 3  # embeddings, blocks, output norm and head
    assert np.array_equal(t["token_embd.weight"], sd["transformer.wte.weight"].numpy())
    assert np.array_equal(t["output.weight"], sd["transformer.wte.weight"].numpy())
    assert np.array_equal(t["blk.1.attn_qkv.bias"], sd["transformer.h.1.attn.c_attn.bias"].numpy())
    assert np.array_equal(t["blk.0.ffn_down.weight"], sd["transformer.h.0.mlp.3.weight"].numpy())
    f = r.fields["tokenizer.ggml.tokens"]
    tokens = [bytes(f.parts[i]).decode() for i in f.data]
    assert tokens == ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "a", "b", "ab", "Ġa"]


def test_vocabulary_size_must_match_recipe(tmp_path):
    make_run(tmp_path)
    (tmp_path / "recipe.json").write_text(json.dumps({"model": {**CFG, "vocab_size": 99}}))
    with pytest.raises(subprocess.CalledProcessError):
        export(tmp_path)


def test_utf8_tokenizer_without_ascii_escaping(tmp_path):
    make_run(tmp_path)
    tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġa": 3}, "merges": [["a", "b"]]}}
    (tmp_path / "tokenizer.json").write_bytes(json.dumps(tok, ensure_ascii=False).encode("utf-8"))
    r = export(tmp_path)
    f = r.fields["tokenizer.ggml.tokens"]
    tokens = [bytes(f.parts[i]).decode() for i in f.data]
    assert "Ġa" in tokens
