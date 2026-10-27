"""CPU-only checks that simple/ovl.py verify compares the chain head to anchor.json."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple"), str(ROOT / "src")]
_spec = importlib.util.spec_from_file_location("ovl_simple_anchor", ROOT / "simple" / "ovl.py")
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)


def link(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def fake_chain(n):
    chain = []
    for i in range(n):
        chain.append({"index": i, "kind": "final" if i == n - 1 else "progress", "control": {"global_step": i},
                      "state": f"{i:064x}", "model": f"{i:064x}", "previous": link(chain[-1]) if chain else None})
    return chain


def args(run, anchor=None, no_anchor=False):
    return argparse.Namespace(run=str(run), data=None, segment=None, random=0, all=False, weights=None, anchor=anchor,
                              no_anchor=no_anchor)


class AnchorHead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.chain = fake_chain(5)
        (self.run / "anchor.json").write_text(json.dumps(
            {"chain_entries": 5, "chain_head_index": 4, "chain_head_sha256": link(self.chain[-1])}))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, chain):
        (self.run / "chain.jsonl").write_text("".join(json.dumps(e) + "\n" for e in chain))

    def verify_ok(self, a):
        # The fake run has no weights or segments, so verify may FAIL on zero checks
        # (see #206/#207). Only the anchor check is under test here.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                ovl.cmd_verify(a)
            except SystemExit as e:
                self.assertNotIn("chain head differs", str(e.code))
        return out.getvalue()

    def test_full_chain_matches_anchor(self):
        self.write(self.chain)
        self.assertIn("chain head MATCH (5 entries)", self.verify_ok(args(self.run)))

    def test_truncated_chain_fails(self):
        self.write(self.chain[:3])  # links are intact, only the tail is missing
        with self.assertRaisesRegex(SystemExit, "chain head differs"):
            ovl.cmd_verify(args(self.run))

    def test_explicit_anchor_path(self):
        self.write(self.chain[:4])
        anchor = self.run / "anchor.json"
        os.rename(anchor, self.run / "release-anchor.json")
        with self.assertRaisesRegex(SystemExit, "not found"):  # no anchor next to the chain
            ovl.cmd_verify(args(self.run))
        with self.assertRaisesRegex(SystemExit, "chain head differs"):
            ovl.cmd_verify(args(self.run, str(self.run / "release-anchor.json")))

    def test_missing_anchor_fails_by_default(self):
        self.write(self.chain)
        os.remove(self.run / "anchor.json")
        with self.assertRaisesRegex(SystemExit, "not found.*--no-anchor"):
            ovl.cmd_verify(args(self.run))

    def test_no_anchor_opt_out_warns_and_continues(self):
        self.write(self.chain[:3])  # would fail the anchor check if one were read
        os.remove(self.run / "anchor.json")
        out = self.verify_ok(args(self.run, no_anchor=True))
        self.assertIn("WARNING: --no-anchor", out)
        self.assertNotIn("chain head MATCH", out)

    def test_anchor_and_no_anchor_are_exclusive(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ("simple", "src", ".")))
        r = subprocess.run([sys.executable, str(ROOT / "simple" / "ovl.py"), "verify", "--run", str(self.run),
                            "--anchor", "a.json", "--no-anchor"], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not allowed with", r.stderr)

    def test_missing_explicit_anchor_fails(self):
        self.write(self.chain)
        with self.assertRaisesRegex(SystemExit, "not found"):
            ovl.cmd_verify(args(self.run, str(self.run / "nope.json")))


if __name__ == "__main__":
    unittest.main()
