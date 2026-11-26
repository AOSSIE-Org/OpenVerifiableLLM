"""Regression tests for ovl.py chat --temperature finite validation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bad_temp", ["nan", "inf", "-inf", "-0.5"])
def test_chat_rejects_nonfinite_or_negative_temperature(tmp_path, bad_temp):
    env = {**os.environ, "OVL_DEVICE": "cpu", "OVL_TIMESTAMP": "0"}
    res = subprocess.run(
        [sys.executable, str(ROOT / "simple" / "ovl.py"), "chat",
         "--run", str(tmp_path), "--tokenizer", str(tmp_path / "tok.json"),
         f"--temperature={bad_temp}", "hello"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert res.returncode != 0
    assert "--temperature must be non-negative and finite" in res.stderr
    assert "Traceback" not in res.stderr
