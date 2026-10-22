"""Regression tests for simple/ovl.py zero-check verification vulnerability.

Tests that:
1. An invocation of `simple/ovl.py verify` with no check options (zero checks performed)
   does not report PASS, exits nonzero, and records result: FAIL in verifications.jsonl.
2. Invoking with `--all` or `--segment` without `--data` does not report PASS and exits nonzero.
3. Legitimate verification paths (e.g. valid matching weights) still pass and exit 0.
4. Mismatched weights fail and exit nonzero.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "simple-run"
PYTHON = sys.executable


class SimpleVerifyZeroCheckTests(unittest.TestCase):
    def _setup_run_dir(self, dest):
        if FIXTURE_DIR.exists() and (FIXTURE_DIR / "chain.jsonl").exists():
            shutil.copytree(FIXTURE_DIR, dest, dirs_exist_ok=True)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            e0 = {"index": 0, "kind": "initial", "state": "0" * 64, "model": "0" * 64, "previous": None}
            h0 = hashlib.sha256(json.dumps(e0, sort_keys=True).encode()).hexdigest()
            e1 = {"index": 1, "kind": "final", "state": "1" * 64, "model": "1" * 64, "previous": h0}
            (dest / "chain.jsonl").write_text(json.dumps(e0) + "\n" + json.dumps(e1) + "\n")

    def _run_verify(self, run_dir, extra_args=None):
        cmd = [PYTHON, str(ROOT / "simple" / "ovl.py"), "verify", "--run", str(run_dir)]
        if extra_args:
            cmd.extend(extra_args)
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{ROOT / 'simple'}{os.pathsep}{ROOT / 'src'}{os.pathsep}{ROOT}"
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return res.returncode, res.stdout, res.stderr

    def test_zero_checks_fails_closed(self):
        """Zero checks invocation must not report PASS, must exit nonzero, and must record FAIL in report."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._setup_run_dir(d)
            code, out, err = self._run_verify(d)

            # 1. Must not report PASS
            self.assertNotIn("verify PASS", out, "Zero-check invocation must not report 'verify PASS'")
            self.assertIn("verify FAIL", out, "Zero-check invocation should report 'verify FAIL'")

            # 2. Must exit nonzero
            self.assertNotEqual(code, 0, "Zero-check invocation must exit with a nonzero code")

            # 3. Records a failure accurately in verification report
            verif_file = d / "verifications.jsonl"
            self.assertTrue(verif_file.exists(), "verifications.jsonl must exist")
            lines = verif_file.read_text().splitlines()
            self.assertTrue(len(lines) > 0, "verifications.jsonl must contain records")
            last_record = json.loads(lines[-1])
            self.assertEqual(last_record.get("result"), "FAIL", "Verification record result must be FAIL")
            self.assertEqual(len(last_record.get("checks", [])), 0, "Checks array must be empty")

    def test_all_flag_without_data_fails_closed(self):
        """Invoking --all without --data performs 0 checks and must fail closed."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._setup_run_dir(d)
            code, out, err = self._run_verify(d, ["--all"])

            self.assertNotIn("verify PASS", out, "--all without --data must not report PASS")
            self.assertIn("verify FAIL", out, "--all without --data must report FAIL")
            self.assertNotEqual(code, 0, "--all without --data must exit nonzero")

            verif_file = d / "verifications.jsonl"
            last_record = json.loads(verif_file.read_text().splitlines()[-1])
            self.assertEqual(last_record.get("result"), "FAIL")

    def test_segment_flag_without_data_fails_closed(self):
        """Invoking --segment without --data must not succeed."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._setup_run_dir(d)
            code, out, err = self._run_verify(d, ["--segment", "0"])

            self.assertNotEqual(code, 0, "--segment without --data must exit nonzero")
            self.assertNotIn("verify PASS", out)

    def test_legitimate_weights_verification_passes(self):
        """Legitimate verification with matching weights passes and exits 0."""
        import torch
        sys.path.insert(0, str(ROOT / "simple"))
        from ovl import model_digest

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._setup_run_dir(d)

            # Create weights matching the terminal entry in chain.jsonl
            sd = {"linear.weight": torch.randn(4, 4)}
            weights_file = d / "weights.pt"
            torch.save(sd, weights_file)
            digest = model_digest(sd)

            chain_lines = (d / "chain.jsonl").read_text().splitlines()
            last_entry = json.loads(chain_lines[-1])
            last_entry["model"] = digest
            chain_lines[-1] = json.dumps(last_entry)
            (d / "chain.jsonl").write_text("\n".join(chain_lines) + "\n")

            code, out, err = self._run_verify(d, ["--weights", str(weights_file)])
            self.assertEqual(code, 0, f"Matching weights should pass. out: {out}, err: {err}")
            self.assertIn("verify PASS: 1 checks", out)

            verif_file = d / "verifications.jsonl"
            last_record = json.loads(verif_file.read_text().splitlines()[-1])
            self.assertEqual(last_record.get("result"), "PASS")
            self.assertEqual(len(last_record.get("checks")), 1)
            self.assertTrue(last_record["checks"][0].get("match"))

    def test_legitimate_weights_mismatch_fails(self):
        """Weights verification with mismatched weights fails and exits nonzero."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._setup_run_dir(d)

            sd = {"mismatch.weight": torch.randn(2, 2)}
            weights_file = d / "wrong_weights.pt"
            torch.save(sd, weights_file)

            code, out, err = self._run_verify(d, ["--weights", str(weights_file)])
            self.assertNotEqual(code, 0, "Mismatched weights must exit nonzero")
            self.assertIn("verify FAIL", out)

            verif_file = d / "verifications.jsonl"
            last_record = json.loads(verif_file.read_text().splitlines()[-1])
            self.assertEqual(last_record.get("result"), "FAIL")
            self.assertEqual(len(last_record.get("checks")), 1)
            self.assertFalse(last_record["checks"][0].get("match"))


if __name__ == "__main__":
    unittest.main()
