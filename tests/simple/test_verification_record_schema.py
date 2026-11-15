"""Regression for AOSSIE-Org/OpenVerifiableLLM#211.

Published release entries in verifications.jsonl use:
  {checker, code, driver, gpu, result: MATCH|MISMATCH, segment: int}

cmd_verify used to append a different local report shape
  {time, environment, chain_head, checks[], result: PASS|FAIL}
to that same file, mixing incompatible schemas.

Expected: verify leaves the published file alone and writes local results to
local-verifications.jsonl.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

PUBLISHED = {
    "checker": "project second pod",
    "code": "ovl.py 202ac6ce",
    "driver": "595.91.07",
    "gpu": "NVIDIA GeForce RTX 5090",
    "result": "MATCH",
    "segment": 0,
}


def _setup_run_dir(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    e0 = {"index": 0, "kind": "initial", "state": "0" * 64, "model": "0" * 64, "previous": None}
    h0 = hashlib.sha256(json.dumps(e0, sort_keys=True).encode()).hexdigest()
    e1 = {"index": 1, "kind": "final", "state": "1" * 64, "model": "1" * 64, "previous": h0}
    (dest / "chain.jsonl").write_text(json.dumps(e0) + "\n" + json.dumps(e1) + "\n")


def _run_verify(run_dir: Path, extra_args=None):
    cmd = [PYTHON, str(ROOT / "simple" / "ovl.py"), "verify", "--run", str(run_dir), "--no-anchor"]
    if extra_args:
        cmd.extend(extra_args)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'simple'}"
    env["OVL_DEVICE"] = "cpu"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return res.returncode, res.stdout, res.stderr


class VerificationRecordSchemaTests(unittest.TestCase):
    def test_verify_does_not_append_local_report_to_published_verifications(self):
        """Published verifications.jsonl must stay in the release schema after verify."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _setup_run_dir(d)
            published_line = json.dumps(PUBLISHED, sort_keys=True) + "\n"
            (d / "verifications.jsonl").write_text(published_line)

            code, out, err = _run_verify(d)
            self.assertNotEqual(code, 0, "zero-check verify should still fail closed")
            self.assertIn("verify FAIL", out)

            after = (d / "verifications.jsonl").read_text()
            self.assertEqual(
                after,
                published_line,
                "verify must not append a local report into published verifications.jsonl",
            )
            lines = [json.loads(l) for l in after.splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["result"], "MATCH")
            self.assertEqual(lines[0]["segment"], 0)
            self.assertNotIn("checks", lines[0])
            self.assertNotIn("environment", lines[0])

    def test_verify_writes_local_report_to_local_verifications_jsonl(self):
        """Local PASS/FAIL reports belong in local-verifications.jsonl."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _setup_run_dir(d)
            (d / "verifications.jsonl").write_text(json.dumps(PUBLISHED, sort_keys=True) + "\n")

            code, out, err = _run_verify(d)
            self.assertNotEqual(code, 0)

            local = d / "local-verifications.jsonl"
            self.assertTrue(local.exists(), "local-verifications.jsonl must be created")
            record = json.loads(local.read_text().splitlines()[-1])
            self.assertEqual(record["result"], "FAIL")
            self.assertIn("checks", record)
            self.assertIn("environment", record)
            self.assertIn("chain_head", record)
            self.assertIn("time", record)
            self.assertNotIn("segment", record)  # published field; local nests under checks
            self.assertNotEqual(record["result"], "MATCH")
            self.assertNotEqual(record["result"], "MISMATCH")

    def test_weights_pass_still_records_only_in_local_file(self):
        """A successful local verify must not touch published verifications.jsonl."""
        import torch

        sys.path.insert(0, str(ROOT / "simple"))
        from ovl import model_digest

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _setup_run_dir(d)
            published_line = json.dumps(PUBLISHED, sort_keys=True) + "\n"
            (d / "verifications.jsonl").write_text(published_line)

            sd = {"linear.weight": torch.randn(4, 4)}
            weights_file = d / "weights.pt"
            torch.save(sd, weights_file)
            digest = model_digest(sd)
            chain_lines = (d / "chain.jsonl").read_text().splitlines()
            last_entry = json.loads(chain_lines[-1])
            last_entry["model"] = digest
            chain_lines[-1] = json.dumps(last_entry)
            (d / "chain.jsonl").write_text("\n".join(chain_lines) + "\n")

            code, out, err = _run_verify(d, ["--weights", str(weights_file)])
            self.assertEqual(code, 0, f"expected PASS; out={out!r} err={err!r}")
            self.assertIn("verify PASS", out)

            self.assertEqual((d / "verifications.jsonl").read_text(), published_line)
            record = json.loads((d / "local-verifications.jsonl").read_text().splitlines()[-1])
            self.assertEqual(record["result"], "PASS")
            self.assertEqual(len(record["checks"]), 1)
            self.assertTrue(record["checks"][0].get("match"))


if __name__ == "__main__":
    unittest.main()
