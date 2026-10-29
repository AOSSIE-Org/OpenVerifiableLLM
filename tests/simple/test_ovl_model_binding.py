"""The chain's `model` digest must be bound to the replayed state (CPU only, tiny model, no data).

`--weights` trusts each entry's `model` digest, so segment verification and checkpoint loading must
check it against the actual weights; otherwise a chain whose `state` replays honestly can point
`model` at weights the training never produced and every check still passes.
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple"), str(ROOT / "src")]
_spec = importlib.util.spec_from_file_location("ovl_simple_binding", ROOT / "simple" / "ovl.py")
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)

TINY = {"attn_impl": "manual", "dropout": 0, "embed_dim": 8, "max_seq_len": 8,
        "num_heads": 2, "num_layers": 1, "vocab_size": 16}


def link(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


class ModelDigestBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        patch = mock.patch.dict(ovl.RECIPE["model"], TINY)
        patch.start()
        self.addCleanup(patch.stop)
        model, opt = ovl.fresh()
        ctl = ovl.initial_control()
        torch.save({"control": ctl, "model": model.state_dict(), "opt": opt.state_dict()}, self.run / "ckpt-00000.pt")
        state, weights = ovl.state_digest(model, opt), ovl.model_digest(model.state_dict())
        first = {"index": 0, "kind": "initial", "control": ctl, "state": state, "model": weights,
                 "previous": None, "time": 0}
        # Entry 1 is what the (stubbed) replay below reaches: same weights, one step later.
        self.chain = [first, {"index": 1, "kind": "progress", "control": {**ctl, "global_step": 1}, "state": state,
                              "model": weights, "previous": link(first), "time": 0}]

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self):
        def replay(data, model, opt, on_boundary, start):  # stands in for training k -> k+1
            return on_boundary(self.chain[1]["kind"], self.chain[1]["control"], opt)
        with mock.patch.object(ovl, "run_phases", replay):
            return ovl.verify_segment(None, self.run, self.chain, 0)

    def test_honest_segment_matches(self):
        self.assertTrue(self.verify())

    def test_segment_with_swapped_model_digest_fails(self):
        # `state` still replays bit for bit; only the digest `--weights` trusts points elsewhere.
        self.chain[1]["model"] = ovl.model_digest({"w": torch.ones(2, 2)})
        self.assertFalse(self.verify())

    def test_checkpoint_with_swapped_model_digest_is_rejected(self):
        self.chain[0]["model"] = ovl.model_digest({"w": torch.ones(2, 2)})
        model, _ = ovl.fresh()
        with self.assertRaises(SystemExit):
            ovl.load_checkpoint(self.run, self.chain[0], model)


if __name__ == "__main__":
    unittest.main()
