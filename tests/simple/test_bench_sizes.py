"""Regression tests for simple/bench_sizes.py (#379)."""
import json
import os
from pathlib import Path
import sys
import threading
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("OVL_DEVICE", "cpu")
os.environ.setdefault("OVL_TIMESTAMP", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import bench_sizes  # noqa: E402
import ovl  # noqa: E402


def _make_prep_data(tmp_path, batches):
    phase = tmp_path / "wikipedia"
    phase.mkdir(parents=True)
    n = batches * ovl.RECIPE["batch_size"] * ovl.RECIPE["context"]
    np.full(n, 5, dtype="<u2").tofile(phase / "tokens.u16")
    np.ones(n, dtype=np.uint8).tofile(phase / "mask.u8")
    (phase / "documents.jsonl").write_text(
        json.dumps({"tokens": n, "offset": 0, "target_start": 0, "targets": n}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _prefetch_workers():
    return [
        t for t in threading.enumerate()
        if getattr(getattr(t, "_target", None), "__qualname__", "") == "prefetch.<locals>.work"
    ]


def test_bench_sizes_closes_prefetch_on_early_break(tmp_path, capsys):
    data_dir = _make_prep_data(tmp_path, batches=70)
    baseline = len(_prefetch_workers())
    saved_model = dict(ovl.RECIPE["model"])
    try:
        with mock.patch.object(bench_sizes, "SIZES", [(64, 1)]), \
             mock.patch.object(ovl, "step", side_effect=lambda m, o, b, c: (None, c["cursor"] + 1, 0)), \
             mock.patch.object(sys, "argv", ["bench_sizes.py", "--data", str(data_dir), "--seconds", "0"]):
            bench_sizes.main()
    finally:
        ovl.RECIPE["model"] = saved_model

    out = capsys.readouterr().out.strip()
    row = json.loads(out)
    assert row["embed_dim"] == 64
    assert row["peak_mem_gb"] == 0.0
    assert len(_prefetch_workers()) == baseline


def test_bench_sizes_short_stream_exits_cleanly_without_crash(tmp_path):
    data_dir = _make_prep_data(tmp_path, batches=5)
    baseline = len(_prefetch_workers())
    saved_model = dict(ovl.RECIPE["model"])
    try:
        with mock.patch.object(bench_sizes, "SIZES", [(64, 1)]), \
             mock.patch.object(ovl, "step", side_effect=lambda m, o, b, c: (None, c["cursor"] + 1, 0)), \
             mock.patch.object(sys, "argv", ["bench_sizes.py", "--data", str(data_dir), "--seconds", "1"]):
            with pytest.raises(SystemExit, match="need > 30"):
                bench_sizes.main()
    finally:
        ovl.RECIPE["model"] = saved_model

    assert len(_prefetch_workers()) == baseline
