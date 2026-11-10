"""CPU-only checks of simple/review_ckpt.py using tiny fake runs (no prepared data, no GPU)."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "simple"), str(Path(__file__).resolve().parent)]

import review_ckpt  # noqa: E402
from test_ovl import fake_chain, tiny_state, write_chain  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
