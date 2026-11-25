"""Regression tests for ovl.py verify --segment and --random bounds validation (#378)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import ovl  # noqa: E402


def _make_run_and_data(tmp_path, entries=2):
    run = tmp_path / "run"
    run.mkdir()
    data = tmp_path / "data"
    for phase in ("wikipedia", "conversation"):
        (data / phase).mkdir(parents=True)
        (data / phase / "stream.json").write_text("{}", encoding="utf-8")
    n = 64 * 512
    np.full(n, 5, dtype="<u2").tofile(data / "wikipedia" / "tokens.u16")
    np.ones(n, dtype=np.uint8).tofile(data / "wikipedia" / "mask.u8")
    (data / "wikipedia" / "documents.jsonl").write_text(
        '{"tokens":100,"offset":0,"target_start":0,"targets":100}\n', encoding="utf-8"
    )
    stream_hash = hashlib.sha256(b"{}").hexdigest()

    chain = []
    for idx in range(entries):
        prev = hashlib.sha256(json.dumps(chain[-1], sort_keys=True).encode()).hexdigest() if chain else None
        kind = "initial" if idx == 0 else ("final" if idx == entries - 1 else "progress")
        entry = {
            "index": idx,
            "kind": kind,
            "previous": prev,
            "state": f"{idx}" * 64,
            "model": f"{idx}" * 64,
            "control": ovl.initial_control(),
        }
        if idx == 0:
            entry["data"] = {"wikipedia": stream_hash, "conversation": stream_hash}
        chain.append(entry)

    (run / "chain.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in chain), encoding="utf-8"
    )
    return run, data


def _run_verify(run, data, extra_args):
    env = {**os.environ, "OVL_DEVICE": "cpu", "OVL_TIMESTAMP": "0"}
    return subprocess.run(
        [sys.executable, str(ROOT / "simple" / "ovl.py"), "verify",
         "--run", str(run), "--data", str(data), "--no-anchor", *extra_args],
        capture_output=True, text=True, env=env, timeout=60,
    )


@pytest.mark.parametrize("bad_seg", ["-1", "1", "99"])
def test_verify_rejects_out_of_range_segment_before_model_init(tmp_path, bad_seg):
    run, data = _make_run_and_data(tmp_path, entries=2)  # valid segment is 0 only
    res = _run_verify(run, data, ["--segment", bad_seg])
    assert res.returncode != 0
    assert "out of range" in res.stderr
    assert "0..0" in res.stderr
    assert "initialization:" not in res.stdout
    assert "Traceback" not in res.stderr


def test_verify_rejects_segment_on_single_entry_chain(tmp_path):
    run, data = _make_run_and_data(tmp_path, entries=1)
    res = _run_verify(run, data, ["--segment", "0"])
    assert res.returncode != 0
    assert "no segments" in res.stderr
    assert "Traceback" not in res.stderr


def test_verify_rejects_negative_random(tmp_path):
    run, data = _make_run_and_data(tmp_path, entries=2)
    res = _run_verify(run, data, ["--random", "-1"])
    assert res.returncode != 0
    assert "--random must be non-negative" in res.stderr
    assert "initialization:" not in res.stdout


def test_verify_accepts_valid_segment_boundary(tmp_path):
    run, data = _make_run_and_data(tmp_path, entries=2)
    res = _run_verify(run, data, ["--segment", "0"])
    # Fails at missing ckpt-00000.pt (not bounds validation), confirming 0 is accepted
    assert "out of range" not in res.stderr
    assert "initialization:" in res.stdout
