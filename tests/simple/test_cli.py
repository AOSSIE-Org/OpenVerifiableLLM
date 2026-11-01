"""The command line on a computer without a GPU: what runs, what stops early, and the messages."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import ovl  # noqa: E402


def verify_args(**kw):
    base = dict(run=None, data=None, segment=None, random=0, all=False, weights=None, anchor=None,
                no_anchor=True, no_timestamps=False)
    return argparse.Namespace(**{**base, **kw})


class RequireGpuTests(unittest.TestCase):
    def test_stops_on_cpu_when_no_device_was_chosen(self):
        with mock.patch.object(ovl, "DEV", "cpu"), mock.patch.dict(os.environ, clear=False) as env:
            env.pop("OVL_DEVICE", None)
            with self.assertRaisesRegex(SystemExit, "Training needs an NVIDIA GPU.*OVL_DEVICE=cpu"):
                ovl.require_gpu("Training")

    def test_runs_on_cpu_when_asked_explicitly(self):
        with mock.patch.object(ovl, "DEV", "cpu"), mock.patch.dict(os.environ, {"OVL_DEVICE": "cpu"}):
            ovl.require_gpu("Training")

    def test_runs_on_a_gpu(self):
        with mock.patch.object(ovl, "DEV", "cuda"), mock.patch.dict(os.environ, clear=False) as env:
            env.pop("OVL_DEVICE", None)
            ovl.require_gpu("Training")

    def test_segment_check_without_a_gpu_stops_before_reading_anything(self):
        with mock.patch.object(ovl, "DEV", "cpu"), mock.patch.dict(os.environ, clear=False) as env:
            env.pop("OVL_DEVICE", None)
            with self.assertRaisesRegex(SystemExit, "Recomputing a segment needs an NVIDIA GPU"):
                ovl.cmd_verify(verify_args(run="/nonexistent", data="/nonexistent", random=1))


class DeviceDefaultTests(unittest.TestCase):
    def test_falls_back_to_cpu_without_a_gpu(self):
        env = {k: v for k, v in os.environ.items() if k != "OVL_DEVICE"}
        code = "import sys; sys.path.insert(0, 'simple'); import torch; torch.cuda.is_available = lambda: False; import ovl; print(ovl.DEV)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "cpu")


class MissingInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / "chain.jsonl").write_text('{"index": 0, "kind": "initial", "state": "s", "model": "m", "previous": null}\n')

    def test_missing_weights_file_is_reported(self):
        with self.assertRaisesRegex(SystemExit, "weights file .*nope.pt not found"):
            ovl.cmd_verify(verify_args(run=str(self.run_dir), weights=str(self.run_dir / "nope.pt")))

    def test_data_directory_without_streams_is_reported(self):
        with self.assertRaisesRegex(SystemExit, "is not a prepared data directory"):
            ovl.cmd_verify(verify_args(run=str(self.run_dir), data=str(self.run_dir)))


class HelpTests(unittest.TestCase):
    def test_verify_options_are_described(self):
        out = subprocess.run([sys.executable, str(ROOT / "simple" / "ovl.py"), "verify", "--help"],
                             capture_output=True, text=True, check=True).stdout
        for words in ("the run directory", "recompute every segment", "this model file"):
            self.assertIn(words, out)


if __name__ == "__main__":
    unittest.main()
