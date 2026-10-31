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
from unittest import mock

import torch

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


def args(run, anchor=None, no_anchor=False, no_timestamps=True):
    # The anchors here are fake and unstamped; AnchorTimestamps below covers the timestamp check.
    return argparse.Namespace(run=str(run), data=None, segment=None, random=0, all=False, weights=None, anchor=anchor,
                              no_anchor=no_anchor, no_timestamps=no_timestamps)


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
        weights = {"w": torch.arange(4.0)}
        chain = fake_chain(3)  # shorter than the anchor, so it would fail the anchor check if one were read
        chain[-1]["model"] = ovl.model_digest(weights)
        self.write(chain)
        torch.save(weights, self.run / "final-model.pt")
        os.remove(self.run / "anchor.json")
        a = args(self.run, no_anchor=True)
        a.weights = str(self.run / "final-model.pt")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ovl.cmd_verify(a)  # a passing check must not exit
        self.assertIn("WARNING: --no-anchor", out.getvalue())
        self.assertNotIn("chain head MATCH", out.getvalue())
        self.assertIn("verify PASS", out.getvalue())
        record = json.loads((self.run / "verifications.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["result"], "PASS")

    def test_no_anchor_opt_out_still_fails_bad_weights(self):
        self.write(fake_chain(3))
        torch.save({"w": torch.arange(4.0)}, self.run / "other.pt")
        os.remove(self.run / "anchor.json")
        a = args(self.run, no_anchor=True)
        a.weights = str(self.run / "other.pt")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            ovl.cmd_verify(a)

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

    def test_file_that_is_not_an_anchor_fails_clearly(self):
        self.write(self.chain)
        for text in ('{"model": {}}', "[1, 2]", "not json"):
            (self.run / "recipe.json").write_text(text)
            with self.assertRaisesRegex(SystemExit, "not an anchor file"):
                ovl.cmd_verify(args(self.run, str(self.run / "recipe.json")))

    def test_anchor_result_is_recorded(self):
        self.write(self.chain)
        with contextlib.redirect_stdout(io.StringIO()):
            ovl.cmd_verify(args(self.run))
        record = json.loads((self.run / "verifications.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["checks"][0]["anchor"], str(self.run / "anchor.json"))
        self.assertTrue(record["checks"][0]["match"])
        self.assertEqual(record["result"], "PASS")


class AnchorTimestamps(unittest.TestCase):
    """A rewritten anchor.json still matches a rewritten chain; only its timestamps catch that."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        chain = fake_chain(5)
        (self.run / "chain.jsonl").write_text("".join(json.dumps(e) + "\n" for e in chain))
        (self.run / "anchor.json").write_text(json.dumps(
            {"chain_entries": 5, "chain_head_index": 4, "chain_head_sha256": link(chain[-1])}))
        for name in ovl.stamp.TSA:
            (self.run / f"anchor.{name}.tsr").write_bytes(b"reply")
        for target, value in (("system_ca", "/ca.pem"), ("freetsa_certs", ("ca", "tsa"))):
            patch = mock.patch.object(ovl.stamp, target, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ovl.cmd_verify(args(self.run, no_timestamps=False, **kw))
        return out.getvalue()

    def test_both_timestamps_are_checked_against_the_anchor_bytes(self):
        with mock.patch.object(ovl.stamp, "verify_reply", return_value=(True, "Oct 20 14:38:43 2026 GMT")) as v:
            out = self.verify()
        self.assertEqual(sorted(c.args[1] for c in v.call_args_list), sorted(ovl.stamp.TSA))
        for c in v.call_args_list:
            self.assertEqual(c.args[0], self.run / f"anchor.{c.args[1]}.tsr")
            self.assertEqual(c.args[2], ["-data", str(self.run / "anchor.json")])
        self.assertIn("anchor timestamp digicert: OK", out)
        record = json.loads((self.run / "verifications.jsonl").read_text().splitlines()[-1])
        self.assertEqual(set(record["checks"][0]["timestamps"]), set(ovl.stamp.TSA))

    def test_timestamp_that_does_not_verify_fails(self):
        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=[(True, "t"), (False, "t")]):
            with self.assertRaisesRegex(SystemExit, "does not verify"):
                self.verify()

    def test_missing_timestamp_fails(self):
        os.remove(self.run / "anchor.freetsa.tsr")
        with mock.patch.object(ovl.stamp, "verify_reply", return_value=(True, "t")):
            with self.assertRaisesRegex(SystemExit, "anchor.freetsa.tsr not found.*--no-timestamps"):
                self.verify()

    def test_certificates_that_cannot_be_downloaded_fail(self):
        with mock.patch.object(ovl.stamp, "freetsa_certs", side_effect=OSError("offline")):
            with self.assertRaisesRegex(SystemExit, "could not download"):
                self.verify()

    def test_missing_openssl_fails(self):
        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=FileNotFoundError("openssl")):
            with self.assertRaisesRegex(SystemExit, "openssl not found"):
                self.verify()

    def test_no_timestamps_opt_out_warns(self):
        with mock.patch.object(ovl.stamp, "verify_reply") as v:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ovl.cmd_verify(args(self.run, no_timestamps=True))
        v.assert_not_called()
        self.assertIn("WARNING: --no-timestamps", out.getvalue())


if __name__ == "__main__":
    unittest.main()
