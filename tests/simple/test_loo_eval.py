"""Small CPU fixtures for evaluation input validation, not released-model replay."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
spec = importlib.util.spec_from_file_location("loo_eval_tested", ROOT / "simple" / "loo_eval.py")
loo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loo)


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    prep, fork, original = (tmp_path / n for n in ("prep", "fork", "original"))
    for p in (prep / "wikipedia", prep / "tokenizer", fork, original):
        p.mkdir(parents=True)
    Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")).save(str(prep / "tokenizer/tokenizer.json"))
    docs = [{"identity": ["wikipedia", str(i)], "tokens": 3000, "offset": i * 3000,
             "targets": 3000, "target_start": i * 3000} for i in range(3)]
    (prep / "wikipedia/documents.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    np.full(9000, 5, dtype="<u2").tofile(prep / "wikipedia/tokens.u16")
    np.ones(9000, dtype="u1").tofile(prep / "wikipedia/mask.u8")
    first = {"index": 0, "previous": None, "kind": "fork",
             "control": {"cursor": 0, "exclude": {"wikipedia": [[0, 3000]]}}}
    last = {"index": 1, "previous": hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
            "kind": "base", "control": {"cursor": 9000}}
    (fork / "chain.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(last) + "\n")
    args = ["loo_eval.py", "--data", str(prep), "--original", str(original), "--fork", str(fork), "--prompt"]
    monkeypatch.setattr(sys, "argv", args)
    return prep, fork, original, args


@pytest.mark.parametrize("count", [0, -1, -100])
def test_nonpositive_counts_fail_before_reading_inputs(tmp_path, monkeypatch, capsys, count):
    missing = str(tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["loo_eval.py", "--data", missing, "--original", missing,
                                    "--fork", missing, "--controls", str(count)])
    with pytest.raises(SystemExit) as error:
        loo.main()
    assert error.value.code == 2
    assert "--controls must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("count", [3, 50])
def test_insufficient_pool_reports_requested_and_available_before_model_load(evaluation, monkeypatch, capsys, count):
    _, _, _, args = evaluation
    monkeypatch.setattr(sys, "argv", args + ["--controls", str(count)])
    with pytest.raises(SystemExit) as error:
        loo.main()
    assert error.value.code == 2
    assert f"--controls requested {count}, but only 2 eligible" in capsys.readouterr().err


def test_empty_pool_is_reported_before_model_load(evaluation, monkeypatch, capsys):
    prep, _, _, args = evaluation
    (prep / "wikipedia/documents.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", args + ["--controls", "1"])
    with pytest.raises(SystemExit) as error:
        loo.main()
    assert error.value.code == 2
    assert "--controls requested 1, but only 0 eligible" in capsys.readouterr().err


@pytest.mark.parametrize("count", [1, 2])
def test_valid_count_keeps_seeded_sampling_and_produces_evaluation(evaluation, monkeypatch, capsys, count):
    _, fork, original, args = evaluation
    recipe = {**loo.ovl.RECIPE, "context": 64,
              "model": {"vocab_size": 16, "embed_dim": 8, "num_heads": 2, "num_layers": 1,
                        "max_seq_len": 64, "dropout": 0, "attn_impl": "manual"}}
    monkeypatch.setattr(loo.ovl, "RECIPE", recipe)
    model = loo.ovl.TinyGPT(**recipe["model"])
    for run in (original, fork):
        torch.save(model.state_dict(), run / "base-model.pt")
    calls = []
    real_loss = loo.doc_loss

    def observe_loss(model, tok, mask, document):
        calls.append(document["identity"][1])
        return real_loss(model, tok, mask, document)

    monkeypatch.setattr(loo, "doc_loss", observe_loss)
    monkeypatch.setattr(sys, "argv", args + ["--controls", str(count)])
    loo.main()
    result = json.loads(capsys.readouterr().out)
    assert result["control_documents"] == count
    assert result["base_weight_relative_distance"] == 0
    expected = loo.random.Random(1234).sample(["1", "2"], count)
    assert calls == (["0"] + expected) * 2
