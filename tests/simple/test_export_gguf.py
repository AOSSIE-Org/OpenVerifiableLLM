"""export_gguf.py writes a llama.cpp `gpt2` file whose tensors and vocabulary match the source."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple")]
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


def test_export_refuses_to_overwrite_the_source(tmp_path):
    make_run(tmp_path)
    source = tmp_path / "conversation-model.pt"
    before = source.read_bytes()
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(source)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "must not overwrite" in r.stderr
    assert source.read_bytes() == before


def test_export_reports_missing_recipe(tmp_path):
    make_run(tmp_path)
    (tmp_path / "recipe.json").unlink()
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "recipe" in r.stderr and "not found" in r.stderr
    assert "Traceback" not in r.stderr


def test_export_reports_invalid_recipe(tmp_path):
    make_run(tmp_path)
    (tmp_path / "recipe.json").write_text("not json", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "recipe" in r.stderr and "is invalid" in r.stderr
    assert "Traceback" not in r.stderr


def test_export_reports_missing_source_checkpoint(tmp_path):
    make_run(tmp_path)
    (tmp_path / "conversation-model.pt").unlink()
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "source checkpoint" in r.stderr and "not found" in r.stderr
    assert "Traceback" not in r.stderr


def test_export_reports_invalid_source_checkpoint(tmp_path):
    make_run(tmp_path)
    (tmp_path / "conversation-model.pt").write_bytes(b"not a valid checkpoint")
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "source checkpoint" in r.stderr and "is invalid" in r.stderr
    assert "Traceback" not in r.stderr


def test_export_reports_missing_tokenizer(tmp_path):
    make_run(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "tokenizer" in r.stderr and "not found" in r.stderr
    assert "Traceback" not in r.stderr


def test_export_reports_invalid_tokenizer(tmp_path):
    make_run(tmp_path)
    (tmp_path / "tokenizer.json").write_text("not json", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "tokenizer" in r.stderr and "is invalid" in r.stderr
    assert "Traceback" not in r.stderr


def run_export(tmp_path, out=None):
    out = out if out is not None else tmp_path / "out.gguf"
    return subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                           "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(out)],
                          capture_output=True, text=True)


def assert_clean_failure(r, *words):
    assert r.returncode != 0
    for word in words:
        assert word in r.stderr
    assert "Traceback" not in r.stderr


def test_export_refuses_the_source_through_another_path(tmp_path):
    make_run(tmp_path)
    source = tmp_path / "conversation-model.pt"
    before = source.read_bytes()
    alias = tmp_path / "sub" / ".." / "conversation-model.pt"
    (tmp_path / "sub").mkdir()
    assert_clean_failure(run_export(tmp_path, alias), "must not overwrite")
    assert source.read_bytes() == before


def test_export_refuses_a_hard_link_to_the_source(tmp_path):
    make_run(tmp_path)
    source = tmp_path / "conversation-model.pt"
    before = source.read_bytes()
    link = tmp_path / "linked.pt"
    try:
        os.link(source, link)
    except OSError:
        pytest.skip("hard links are not supported here")
    assert_clean_failure(run_export(tmp_path, link), "must not overwrite")
    assert source.read_bytes() == before


def test_export_reports_missing_recipe_field(tmp_path):
    make_run(tmp_path)
    cfg = {k: v for k, v in CFG.items() if k != "num_heads"}
    (tmp_path / "recipe.json").write_text(json.dumps({"model": cfg}), encoding="utf-8")
    assert_clean_failure(run_export(tmp_path), "model.num_heads")


def test_export_reports_recipe_without_model_section(tmp_path):
    make_run(tmp_path)
    (tmp_path / "recipe.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    assert_clean_failure(run_export(tmp_path), "no model section")


def test_export_reports_checkpoint_missing_tensors(tmp_path):
    sd = make_run(tmp_path)
    del sd["transformer.h.1.mlp.3.bias"]
    torch.save(sd, tmp_path / "conversation-model.pt")
    assert_clean_failure(run_export(tmp_path), "missing tensors", "transformer.h.1.mlp.3.bias")
    assert not (tmp_path / "out.gguf").exists()


def test_export_reports_tokenizer_without_merges(tmp_path):
    make_run(tmp_path)
    tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġa": 3}}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    assert_clean_failure(run_export(tmp_path), "model.vocab and model.merges")


def test_export_reports_non_integer_token_ids(tmp_path):
    make_run(tmp_path)
    tok = {"model": {"type": "BPE", "vocab": {"a": "0", "b": 1, "ab": 2, "Ġa": 3}, "merges": [["a", "b"]]}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    assert_clean_failure(run_export(tmp_path), "integer ids")


def test_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    make_run(tmp_path)
    out = tmp_path / "out.gguf"
    out.write_bytes(b"previous good export")
    sys.path.insert(0, str(ROOT / "simple"))
    import export_gguf

    def fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(export_gguf.gguf.GGUFWriter, "write_tensors_to_file", fail)
    monkeypatch.setattr(sys, "argv", ["export_gguf.py", "--run", str(tmp_path),
                                      "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(out)])
    with pytest.raises(SystemExit) as exit_info:
        export_gguf.main()
    assert "could not write" in str(exit_info.value) and "disk full" in str(exit_info.value)
    assert out.read_bytes() == b"previous good export"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_successful_export_leaves_no_temporary_file(tmp_path):
    make_run(tmp_path)
    r = run_export(tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out.gguf").is_file()
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_export_reports_non_contiguous_tokenizer_ids(tmp_path):
    make_run(tmp_path)
    tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 2, "ab": 3, "Ġa": 4}, "merges": [["a", "b"]]}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_gguf.py"), "--run", str(tmp_path),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(tmp_path / "out.gguf")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "tokenizer ids must be contiguous" in r.stderr
    assert "Traceback" not in r.stderr
