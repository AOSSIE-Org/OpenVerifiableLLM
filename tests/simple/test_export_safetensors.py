"""export_safetensors.py keeps every weight, and verify matches a .safetensors file like the .pt file."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

safetensors = pytest.importorskip("safetensors.torch")
os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple")]
from model import TinyGPT  # noqa: E402

_spec = importlib.util.spec_from_file_location("ovl_simple_safetensors", ROOT / "simple" / "ovl.py")
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)

CFG = {"attn_impl": "manual", "dropout": 0, "embed_dim": 16, "max_seq_len": 8, "num_heads": 2,
       "num_layers": 2, "vocab_size": 9}


def link(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def make_run(run):
    torch.manual_seed(0)
    model = TinyGPT(**CFG)
    model.lm_head.weight = model.transformer["wte"].weight  # tied, as in training
    sd = model.state_dict()
    torch.save(sd, run / "conversation-model.pt")
    chain = []
    for i, kind in enumerate(("initial", "final")):
        chain.append({"index": i, "kind": kind, "control": {"global_step": i}, "state": f"{i:064x}",
                      "model": ovl.model_digest(sd) if kind == "final" else "0" * 64,
                      "previous": link(chain[-1]) if chain else None})
    (run / "chain.jsonl").write_text("".join(json.dumps(e) + "\n" for e in chain))
    return sd


def export(run):
    out = run / "conversation.safetensors"
    subprocess.run([sys.executable, str(ROOT / "simple" / "export_safetensors.py"), "--run", str(run),
                    "--out", str(out)], check=True, capture_output=True)
    return out


def verify(run, weights):
    a = argparse.Namespace(run=str(run), data=None, weights=str(weights), segment=None, random=0, all=False,
                           anchor=None, no_anchor=True, no_timestamps=True)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ovl.cmd_verify(a)
    return out.getvalue()


def test_export_keeps_every_weight_and_records_the_source(tmp_path):
    sd = make_run(tmp_path)
    out = export(tmp_path)
    loaded = safetensors.load_file(out)
    assert set(loaded) == set(sd)
    assert all(torch.equal(loaded[k], sd[k]) for k in sd)
    with safetensors.safe_open(out, "pt") as f:
        meta = json.loads(f.metadata()["ovl"])
    assert meta["model_digest"] == ovl.model_digest(sd)
    assert meta["source_sha256"] == hashlib.sha256((tmp_path / "conversation-model.pt").read_bytes()).hexdigest()


def test_export_is_byte_identical_every_time(tmp_path):
    make_run(tmp_path)
    first = export(tmp_path).read_bytes()
    assert all(export(tmp_path).read_bytes() == first for _ in range(3))


def test_export_refuses_to_overwrite_the_source(tmp_path):
    make_run(tmp_path)
    source = tmp_path / "conversation-model.pt"
    before = source.read_bytes()
    r = subprocess.run([sys.executable, str(ROOT / "simple" / "export_safetensors.py"), "--run", str(tmp_path),
                        "--out", str(source)], capture_output=True, text=True)
    assert r.returncode != 0 and "must not overwrite" in r.stderr
    assert source.read_bytes() == before


def test_verify_matches_the_safetensors_file(tmp_path):
    make_run(tmp_path)
    assert "MATCH chain entry 1" in verify(tmp_path, export(tmp_path))


def test_verify_rejects_a_changed_safetensors_file(tmp_path):
    make_run(tmp_path)
    out = export(tmp_path)
    changed = safetensors.load_file(out)
    name = next(k for k, v in changed.items() if v.is_floating_point())
    changed[name][(0,) * changed[name].ndim] += 1e-6
    safetensors.save_file(changed, out)
    with pytest.raises(SystemExit):
        verify(tmp_path, out)
