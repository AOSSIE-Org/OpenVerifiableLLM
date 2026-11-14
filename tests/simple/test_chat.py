"""Tests for simple/ovl.py chat command."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

import torch  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import ovl  # noqa: E402
ovl.DEV = "cpu"


class ChatCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.tok_path = self.run / "tokenizer.json"
        Tokenizer(models.BPE()).save(str(self.tok_path))
        model = ovl.TinyGPT(**ovl.RECIPE["model"])
        torch.save(model.state_dict(), self.run / "conversation-model.pt")

    def tearDown(self):
        self.tmp.cleanup()

    def test_temperature_zero_uses_greedy_decoding(self):
        a = argparse.Namespace(
            run=str(self.run), tokenizer=str(self.tok_path), prompt="Hello",
            model="conversation", tokens=1, temperature=0.0, seed=0,
        )
        vocab_size = ovl.RECIPE["model"]["vocab_size"]
        controlled_logits = torch.zeros(1, 10, vocab_size)
        target_token = 12
        controlled_logits[0, -1, target_token] = 99.0

        with mock.patch("torch.multinomial", side_effect=AssertionError("torch.multinomial must not be called when temperature=0")):
            with mock.patch.object(ovl.TinyGPT, "forward", return_value=controlled_logits):
                with mock.patch.object(Tokenizer, "decode", return_value="out") as mock_decode:
                    ovl.cmd_chat(a)
                    self.assertEqual(mock_decode.call_args[0][0], [target_token - ovl.OFFSET])

    def test_negative_temperature_exits_cleanly(self):
        a = argparse.Namespace(
            run=str(self.run), tokenizer=str(self.tok_path), prompt="Hello",
            model="conversation", tokens=5, temperature=-0.5, seed=0,
        )
        with self.assertRaisesRegex(SystemExit, "--temperature must be non-negative"):
            ovl.cmd_chat(a)


if __name__ == "__main__":
    unittest.main()
