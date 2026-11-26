"""Regression tests for verify_segment() fork exclusion validation."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch

os.environ.setdefault("OVL_DEVICE", "cpu")
os.environ.setdefault("OVL_TIMESTAMP", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import ovl  # noqa: E402


def _make_fork_chain(tmp_path, exclude_obj):
    model, opt = ovl.fresh()
    sd = model.state_dict()
    opt_sd = opt.state_dict()
    m_dig = ovl.model_digest(sd)
    s_dig = ovl.state_digest(model, opt)

    ctl0 = ovl.initial_control()
    ctl0["cursor"] = 100
    e0 = {
        "index": 0,
        "kind": "initial",
        "control": ctl0,
        "state": s_dig,
        "model": m_dig,
        "previous": None,
    }
    h0 = hashlib.sha256(json.dumps(e0, sort_keys=True).encode()).hexdigest()

    ctl1 = {**ctl0, "exclude": exclude_obj}
    e1 = {
        "index": 1,
        "kind": "fork",
        "control": ctl1,
        "state": s_dig,
        "model": m_dig,
        "previous": h0,
    }
    torch.save({"control": ctl0, "model": sd, "opt": opt_sd}, tmp_path / "ckpt-00000.pt")
    torch.save({"control": ctl1, "model": sd, "opt": opt_sd}, tmp_path / "ckpt-00001.pt")
    return [e0, e1]


def test_verify_segment_fork_accepts_valid_exclusion(tmp_path):
    chain = _make_fork_chain(tmp_path, {"wikipedia": [[150, 25]]})
    assert ovl.verify_segment(None, tmp_path, chain, 0) is True


@pytest.mark.parametrize("bad_exclude", [
    {},
    {"wikipedia": []},
    {"wikipedia": None},
    {"conversation": [[150, 25]]},
    {"wikipedia": [[150, 0]]},
    {"wikipedia": [[150, -5]]},
    {"wikipedia": [[150, True]]},
    {"wikipedia": [[50, 25]]},
])
def test_verify_segment_fork_rejects_invalid_exclusion(tmp_path, bad_exclude):
    chain = _make_fork_chain(tmp_path, bad_exclude)
    assert ovl.verify_segment(None, tmp_path, chain, 0) is False
