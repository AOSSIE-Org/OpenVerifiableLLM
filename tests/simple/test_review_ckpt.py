"""CPU-only checks of simple/review_ckpt.py using tiny fake runs (no prepared data, no GPU)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "simple"), str(Path(__file__).resolve().parent)]

import torch  # noqa: E402
import review_ckpt  # noqa: E402
from test_ovl import fake_chain, tiny_state, write_chain  # noqa: E402

ovl = review_ckpt.ovl


def run_main(run):
    """Run review_ckpt.main() on RUN and return (exit code, stdout)."""
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["review_ckpt.py", str(run)]), contextlib.redirect_stdout(out):
        try:
            review_ckpt.main()
        except SystemExit as e:
            return e.code, out.getvalue()
    raise AssertionError("main() returned without exiting")


class NothingToReviewTests(unittest.TestCase):
    def test_valid_chain_but_no_files_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_chain(tmp, fake_chain([("initial", tiny_state(0)), ("progress", tiny_state(1)),
                                         ("base", tiny_state(2)), ("final", tiny_state(3))]))
            code, out = run_main(tmp)
        self.assertEqual(out, "")
        self.assertNotIn(code, (0, None))
        self.assertIn("nothing to review", str(code))


class CheckpointReviewTests(unittest.TestCase):
    """A real (seeded, untrained) boundary checkpoint: honest input passes, tampered input fails."""

    @classmethod
    def setUpClass(cls):
        model, opt = ovl.fresh()
        cls.ctl = ovl.initial_control()
        cls.entry = {"index": 0, "kind": "initial", "control": cls.ctl, "state": ovl.state_digest(model, opt),
                     "model": ovl.model_digest(model.state_dict()), "previous": None, "time": 0}
        cls.ck = {"control": cls.ctl, "model": model.state_dict(), "opt": opt.state_dict()}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def review(self, ck=None, entry=None):
        torch.save(ck or self.ck, self.run / "ckpt-00000.pt")
        write_chain(self.run, [entry or self.entry])
        code, out = run_main(self.run)
        return code, json.loads(out)

    def test_honest_checkpoint_passes(self):
        code, result = self.review()
        self.assertEqual((code, result["result"], result["problems"]), (0, "PASS", []))

    def test_model_file_alone_is_reviewed_and_passes(self):
        torch.save(self.ck["model"], self.run / "base-model.pt")
        write_chain(self.run, [{**self.entry, "kind": "base"}])
        code, out = run_main(self.run)
        self.assertEqual(code, 0)
        self.assertEqual([json.loads(line)["result"] for line in out.splitlines()], ["PASS"])

    def test_model_file_with_altered_weights_fails(self):
        sd = {k: v.clone() for k, v in self.ck["model"].items()}
        sd[next(iter(sd))].add_(1.0)
        torch.save(sd, self.run / "base-model.pt")
        write_chain(self.run, [{**self.entry, "kind": "base"}])
        code, out = run_main(self.run)
        result = json.loads(out)
        self.assertEqual((code, result["result"]), (1, "FAIL"))
        self.assertIn("weights differ from the committed chain entry", result["problems"])

    def test_extra_top_level_key_fails(self):
        code, result = self.review(ck={**self.ck, "extra": torch.zeros(1)})
        self.assertEqual((code, result["result"]), (1, "FAIL"))
        self.assertTrue(any(p.startswith("top-level keys") for p in result["problems"]))

    def test_control_that_differs_from_the_chain_fails(self):
        entry = {**self.entry, "control": {**self.ctl, "global_step": 1}}
        code, result = self.review(entry=entry)
        self.assertEqual((code, result["result"]), (1, "FAIL"))
        self.assertIn("control differs from chain", result["problems"])

    def test_uncovered_checkpoint_fails(self):
        torch.save(self.ck, self.run / "ckpt-00000.pt")
        torch.save(self.ck, self.run / "ckpt-00001.pt")
        write_chain(self.run, [self.entry])
        code, out = run_main(self.run)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(code, 1)
        uncovered = next((l for l in lines if l["file"] == "ckpt-00001.pt"), None)
        self.assertIsNotNone(uncovered)
        self.assertEqual(uncovered["result"], "FAIL")
        self.assertIn("file not covered by any chain entry", uncovered["problems"])

    def test_leftover_ckpt_tmp_fails(self):
        torch.save(self.ck, self.run / "ckpt-00000.pt")
        (self.run / "ckpt.tmp").write_bytes(b"leftover temporary data")
        write_chain(self.run, [self.entry])
        code, out = run_main(self.run)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(code, 1)
        uncovered = next((l for l in lines if l["file"] == "ckpt.tmp"), None)
        self.assertIsNotNone(uncovered)
        self.assertEqual(uncovered["result"], "FAIL")
        self.assertIn("file not covered by any chain entry", uncovered["problems"])

    def test_uncovered_base_model_fails(self):
        torch.save(self.ck, self.run / "ckpt-00000.pt")
        torch.save(self.ck["model"], self.run / "base-model.pt")
        write_chain(self.run, [self.entry])
        code, out = run_main(self.run)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(code, 1)
        uncovered = next((l for l in lines if l["file"] == "base-model.pt"), None)
        self.assertIsNotNone(uncovered)
        self.assertEqual(uncovered["result"], "FAIL")
        self.assertIn("file not covered by any chain entry", uncovered["problems"])

    def test_uncovered_conversation_model_fails(self):
        torch.save(self.ck, self.run / "ckpt-00000.pt")
        torch.save(self.ck["model"], self.run / "conversation-model.pt")
        write_chain(self.run, [self.entry])
        code, out = run_main(self.run)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(code, 1)
        uncovered = next((l for l in lines if l["file"] == "conversation-model.pt"), None)
        self.assertIsNotNone(uncovered)
        self.assertEqual(uncovered["result"], "FAIL")
        self.assertIn("file not covered by any chain entry", uncovered["problems"])

    def test_uncovered_arbitrary_pt_file_fails(self):
        torch.save(self.ck, self.run / "ckpt-00000.pt")
        (self.run / "arbitrary.pt").write_bytes(b"not a valid checkpoint")
        write_chain(self.run, [self.entry])
        code, out = run_main(self.run)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(code, 1)
        uncovered = next((l for l in lines if l["file"] == "arbitrary.pt"), None)
        self.assertIsNotNone(uncovered)
        self.assertEqual(uncovered["result"], "FAIL")
        self.assertIn("file not covered by any chain entry", uncovered["problems"])

    def test_uncovered_file_reported_even_if_checkpoint_validation_fails(self):
        bad_ck = {**self.ck, "model": {k: v.clone() for k, v in self.ck["model"].items()}}
        bad_ck["model"][next(iter(bad_ck["model"]))].add_(1.0)
        torch.save(bad_ck, self.run / "ckpt-00000.pt")
        (self.run / "arbitrary.pt").write_bytes(b"extra")
        write_chain(self.run, [self.entry])
        code, out = run_main(self.run)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(code, 1)
        uncovered = next((l for l in lines if l["file"] == "arbitrary.pt"), None)
        self.assertIsNotNone(uncovered)
        self.assertEqual(uncovered["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
