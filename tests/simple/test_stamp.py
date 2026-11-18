"""Unit tests for simple/stamp.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


class StampMainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_stamp(self, args):
        cmd = [PYTHON, str(ROOT / "simple" / "stamp.py"), str(self.run)] + args
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr

    def test_missing_chain_file_fails(self):
        code, out, err = self._run_stamp(["--check"])
        self.assertNotEqual(code, 0)
        self.assertIn("not found", err)

    def test_empty_chain_fails_closed(self):
        (self.run / "chain.jsonl").write_text("")
        code, out, err = self._run_stamp(["--check"])
        self.assertNotEqual(code, 0)
        self.assertIn("has no chain entries", err)

    def test_whitespace_only_chain_fails_closed(self):
        (self.run / "chain.jsonl").write_text("   \n\n  \n")
        code, out, err = self._run_stamp(["--check"])
        self.assertNotEqual(code, 0)
        self.assertIn("has no chain entries", err)

    def test_malformed_json_fails_with_line_number(self):
        valid = json.dumps({"index": 0, "kind": "initial"})
        (self.run / "chain.jsonl").write_text(f"{valid}\n{{bad json\n")
        code, out, err = self._run_stamp(["--check"])
        self.assertNotEqual(code, 0)
        self.assertIn("line 2 is not valid JSON", err)


if __name__ == "__main__":
    unittest.main()
