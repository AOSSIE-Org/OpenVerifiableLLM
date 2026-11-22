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
from types import SimpleNamespace
from unittest import mock

import torch

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple")]
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


def reply_with_imprint(digest_hex, params=b"\x05\x00"):
    """A reply carrying a SHA-256 message imprint, as a real TSR does.

    The imprint is the AlgorithmIdentifier (sha256, with or without NULL parameters) followed by
    the OCTET STRING bytes in digest_hex (normally 32), as stamp.py builds for a timestamp request.
    """
    return ovl.stamp.SHA256_OID + params + b"\x04\x20" + bytes.fromhex(digest_hex)


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
        record = json.loads((self.run / "local-verifications.jsonl").read_text().splitlines()[-1])
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
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ("simple",)))
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
        record = json.loads((self.run / "local-verifications.jsonl").read_text().splitlines()[-1])
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

    def stamped(self, original):
        """A stand-in for openssl that accepts only the bytes that were stamped."""
        return lambda tsr, name, imprint, certs: (Path(imprint[1]).read_bytes() == original, "Oct 20 14:38:43 2026 GMT")

    def test_both_timestamps_are_checked_against_the_anchor_bytes(self):
        original = (self.run / "anchor.json").read_bytes()
        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=self.stamped(original)) as v:
            out = self.verify()
        self.assertEqual(sorted(c.args[1] for c in v.call_args_list), sorted(ovl.stamp.TSA))
        for c in v.call_args_list:
            self.assertEqual(c.args[0], self.run / f"anchor.{c.args[1]}.tsr")
            self.assertEqual(c.args[2][0], "-data")
        self.assertIn("anchor timestamp digicert: OK", out)
        record = json.loads((self.run / "local-verifications.jsonl").read_text().splitlines()[-1])
        self.assertEqual(set(record["checks"][0]["timestamps"]), set(ovl.stamp.TSA))

    def test_edited_anchor_fails_its_timestamps(self):
        # Same chain head, so the head check passes; any other byte change breaks the stamp.
        original = (self.run / "anchor.json").read_bytes()
        edited = {**json.loads(original), "note": "edited"}
        (self.run / "anchor.json").write_text(json.dumps(edited))
        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=self.stamped(original)):
            with self.assertRaisesRegex(SystemExit, "does not verify"):
                self.verify()

    def test_libressl_ca_failure_is_not_reported_as_anchor_change(self):
        """issue #364: a CA-chain failure must not be blamed on a changed anchor.

        The reply's imprint still matches anchor.json, so it is openssl that failed, not the
        download. verify must fail closed, but say why instead of alleging tampering.
        """
        original = (self.run / "anchor.json").read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        (self.run / "anchor.digicert.tsr").write_bytes(reply_with_imprint(digest))
        ca_error = "Verify error:unable to get local issuer certificate"

        def openssl_stub(tsr, name, imprint, certs):
            if name == "digicert":
                return False, "Oct 20 14:38:43 2026 GMT"
            return True, "Oct 20 14:38:43 2026 GMT"

        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=openssl_stub), \
             mock.patch.object(ovl.stamp, "verify_failure_detail", return_value=ca_error), \
             mock.patch.object(ovl.stamp, "is_libressl", return_value=True):
            with self.assertRaises(SystemExit) as cm:
                self.verify()
        message = str(cm.exception)
        self.assertNotIn("the anchor was changed", message)
        self.assertIn(ca_error, message)
        self.assertIn("OVL_OPENSSL", message)

    def test_genuine_anchor_change_still_fails_as_tampering(self):
        """A reply whose imprint is for different bytes must still be reported as tampering."""
        original = (self.run / "anchor.json").read_bytes()
        (self.run / "anchor.digicert.tsr").write_bytes(reply_with_imprint(hashlib.sha256(original).hexdigest()))
        (self.run / "anchor.json").write_text(json.dumps({**json.loads(original), "note": "edited"}))

        def openssl_stub(tsr, name, imprint, certs):
            if name == "digicert":
                return False, "Oct 20 14:38:43 2026 GMT"
            return True, "Oct 20 14:38:43 2026 GMT"

        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=openssl_stub), \
             mock.patch.object(ovl.stamp, "verify_failure_detail", return_value="digest verify failure"):
            with self.assertRaisesRegex(SystemExit, "the anchor was changed after it was stamped"):
                self.verify()

    def test_absent_parameters_imprint_still_detects_a_changed_anchor(self):
        """RFC 5754 allows no NULL params; a real anchor change must still be reported as tampering."""
        original = (self.run / "anchor.json").read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        (self.run / "anchor.digicert.tsr").write_bytes(reply_with_imprint(digest, params=b""))
        (self.run / "anchor.json").write_text(json.dumps({**json.loads(original), "note": "edited"}))

        def openssl_stub(tsr, name, imprint, certs):
            if name == "digicert":
                return False, "Oct 20 14:38:43 2026 GMT"
            return True, "Oct 20 14:38:43 2026 GMT"

        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=openssl_stub), \
             mock.patch.object(ovl.stamp, "verify_failure_detail", return_value="digest verify failure"):
            with self.assertRaisesRegex(SystemExit, "the anchor was changed after it was stamped"):
                self.verify()

    def test_truncated_reply_is_not_reported_as_anchor_change(self):
        """A marker with no digest must fail closed with the openssl detail, not allege tampering."""
        (self.run / "anchor.digicert.tsr").write_bytes(reply_with_imprint(""))
        ca_error = "Verify error:unable to get local issuer certificate"

        def openssl_stub(tsr, name, imprint, certs):
            if name == "digicert":
                return False, "Oct 20 14:38:43 2026 GMT"
            return True, "Oct 20 14:38:43 2026 GMT"

        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=openssl_stub), \
             mock.patch.object(ovl.stamp, "verify_failure_detail", return_value=ca_error):
            with self.assertRaises(SystemExit) as cm:
                self.verify()
        message = str(cm.exception)
        self.assertNotIn("the anchor was changed", message)
        self.assertIn(ca_error, message)

    def test_partial_imprint_is_not_reported_as_anchor_change(self):
        """Ten digest bytes are not an imprint; slicing must not produce a short hex comparison."""
        original = (self.run / "anchor.json").read_bytes()
        partial = hashlib.sha256(original).hexdigest()[:20]  # 10 of the 32 bytes
        (self.run / "anchor.digicert.tsr").write_bytes(reply_with_imprint(partial))
        ca_error = "Verify error:unable to get local issuer certificate"

        def openssl_stub(tsr, name, imprint, certs):
            if name == "digicert":
                return False, "Oct 20 14:38:43 2026 GMT"
            return True, "Oct 20 14:38:43 2026 GMT"

        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=openssl_stub), \
             mock.patch.object(ovl.stamp, "verify_failure_detail", return_value=ca_error):
            with self.assertRaises(SystemExit) as cm:
                self.verify()
        message = str(cm.exception)
        self.assertNotIn("the anchor was changed", message)
        self.assertIn(ca_error, message)

    def test_timestamps_check_the_bytes_the_head_was_compared_with(self):
        # Swapping anchor.json after it was read must not change what the timestamps check.
        original = (self.run / "anchor.json").read_bytes()

        def swap_then_check(tsr, name, imprint, certs):
            (self.run / "anchor.json").write_text("swapped")
            return self.stamped(original)(tsr, name, imprint, certs)
        with mock.patch.object(ovl.stamp, "verify_reply", side_effect=swap_then_check):
            self.assertIn("anchor timestamp freetsa: OK", self.verify())

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


class ReplyVerification(unittest.TestCase):
    """The openssl verdict itself: stdout alone must not be enough to pass."""

    def _verify(self, returncode, stdout):
        with mock.patch.object(ovl.stamp, "system_ca", return_value="/ca.pem"), \
             mock.patch.object(ovl.stamp.subprocess, "run") as run:
            run.side_effect = [
                SimpleNamespace(returncode=returncode, stdout=stdout, stderr=""),
                SimpleNamespace(returncode=0, stdout="Time stamp: Oct 20 14:38:43 2026 GMT\n", stderr=""),
            ]
            return ovl.stamp.verify_reply("reply.tsr", "digicert", ["-data", "anchor.json"], ("ca", "tsa"))

    def test_nonzero_exit_fails_even_with_ok_on_stdout(self):
        ok, when = self._verify(1, "Verification: OK\n")
        self.assertFalse(ok)
        self.assertEqual(when, "Oct 20 14:38:43 2026 GMT")

    def test_zero_exit_with_ok_on_stdout_passes(self):
        ok, when = self._verify(0, "Verification: OK\n")
        self.assertTrue(ok)
        self.assertEqual(when, "Oct 20 14:38:43 2026 GMT")

    def test_reply_imprint_accepts_both_parameter_forms_and_rejects_short_digests(self):
        """Both RFC 5754 forms yield the digest; a missing or partial digest yields None."""
        digest = hashlib.sha256(b"anchor").hexdigest()
        raw = bytes.fromhex(digest)
        oid = ovl.stamp.SHA256_OID
        cases = (
            (oid + b"\x05\x00\x04\x20" + raw, digest),
            (oid + b"\x04\x20" + raw, digest),
            (oid + b"\x05\x00\x04\x20" + raw[:10], None),
            (oid + b"\x05\x00\x04\x20", None),
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reply.tsr"
            for blob, expected in cases:
                with self.subTest(blob=blob.hex()):
                    path.write_bytes(blob)
                    self.assertEqual(ovl.stamp.reply_imprint(path), expected)


if __name__ == "__main__":
    unittest.main()
