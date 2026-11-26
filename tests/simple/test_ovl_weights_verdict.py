import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple")]

import ovl  # noqa: E402

CHAIN = [{"index": i, "kind": kind} for i, kind in enumerate(["initial", "progress", "base", "final"])]


def test_released_entry_matches():
    assert ovl.weights_verdict(CHAIN, [3], True) == "MATCH chain entry 3"


def test_progress_entry_is_reported_as_in_the_chain_but_not_released():
    message = ovl.weights_verdict(CHAIN, [1], False)
    assert "NOT in the chain" not in message
    assert "chain entry 1 (progress)" in message


def test_unknown_weights_are_not_in_the_chain():
    assert ovl.weights_verdict(CHAIN, [], False) == "NOT in the chain"


def test_load_weights_rejects_non_state_dict(tmp_path):
    import pytest
    import torch

    for name, obj in [("ckpt.pt", {"model": {"w": torch.zeros(1)}, "step": 3}), ("empty.pt", {}), ("list.pt", [torch.zeros(1)])]:
        path = tmp_path / name
        torch.save(obj, path)
        with pytest.raises(SystemExit) as exc:
            ovl.load_weights(path)
        assert "is not a model state dict" in str(exc.value)


def test_load_weights_accepts_state_dict(tmp_path):
    import torch

    path = tmp_path / "ok.pt"
    torch.save({"w": torch.ones(2)}, path)
    assert torch.equal(ovl.load_weights(path)["w"], torch.ones(2))
