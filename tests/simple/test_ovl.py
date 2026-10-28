"""CPU-only checks of simple/ovl.py using tiny fake chains (no prepared data, no GPU)."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple"), str(ROOT / "src")]
_spec = importlib.util.spec_from_file_location("ovl_simple", ROOT / "simple" / "ovl.py")
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)

from model import TinyGPT  # noqa: E402


def link(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def fake_chain(models):
    """One entry per (kind, state_dict); hash-linked exactly as cmd_train writes them."""
    chain, ctl = [], ovl.initial_control()
    for i, (kind, sd) in enumerate(models):
        chain.append({"index": i, "kind": kind, "control": {**ctl, "global_step": i},
                      "state": f"{i:064x}", "model": ovl.model_digest(sd),
                      "previous": link(chain[-1]) if chain else None, "time": 0})
    return chain


def write_chain(run, chain):
    (Path(run) / "chain.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in chain))


def tiny_state(seed):
    torch.manual_seed(seed)
    return {"w": torch.randn(4, 4), "b": torch.randn(4)}


class ChainLinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.chain = fake_chain([("initial", tiny_state(0)), ("progress", tiny_state(1)),
                                 ("base", tiny_state(2)), ("final", tiny_state(3))])

    def tearDown(self):
        self.tmp.cleanup()

    def test_intact_chain_reads(self):
        write_chain(self.run, self.chain)
        self.assertEqual(ovl.read_chain(self.run), self.chain)

    def test_tampered_entry_breaks_the_next_link(self):
        self.chain[1]["state"] = "f" * 64
        write_chain(self.run, self.chain)
        with self.assertRaisesRegex(SystemExit, "chain link broken at entry 2"):
            ovl.read_chain(self.run)

    def test_reordered_entries_fail(self):
        self.chain[1], self.chain[2] = self.chain[2], self.chain[1]
        write_chain(self.run, self.chain)
        with self.assertRaisesRegex(SystemExit, "chain link broken at entry 1"):
            ovl.read_chain(self.run)

    def test_dropped_entry_fails(self):
        del self.chain[1]
        write_chain(self.run, self.chain)
        with self.assertRaisesRegex(SystemExit, "chain link broken at entry 1"):
            ovl.read_chain(self.run)

    def test_relinked_entry_with_wrong_index_fails(self):
        self.chain[1]["index"] = 5
        for i in range(2, len(self.chain)):
            self.chain[i]["previous"] = link(self.chain[i - 1])
        write_chain(self.run, self.chain)
        with self.assertRaisesRegex(SystemExit, "chain link broken at entry 1"):
            ovl.read_chain(self.run)


class LearningRateTests(unittest.TestCase):
    def test_depends_only_on_step_and_phase(self):
        a = {"phase": "wikipedia", "global_step": 1234, "phase_step": 1234, "cursor": 0, "epoch": 0}
        b = {"phase": "wikipedia", "global_step": 1234, "phase_step": 7, "cursor": 99, "epoch": 3,
             "pos": [5, 6], "exclude": {"wikipedia": [[1, 2]]}}
        self.assertEqual(ovl.learning_rate(a), ovl.learning_rate(b))

    def test_schedule_shape(self):
        s = ovl.RECIPE["schedule"]
        peak, floor, warm, n = float(s["peak"]), float(s["floor"]), s["warmup"], s["wikipedia_updates"]
        lr = lambda k: ovl.learning_rate({"phase": "wikipedia", "global_step": k})  # noqa: E731
        self.assertEqual(lr(0), peak / warm)
        self.assertEqual(lr(warm - 1), peak)
        self.assertAlmostEqual(lr(warm), peak)
        self.assertAlmostEqual(lr(n), floor)
        self.assertAlmostEqual(lr(10 * n), floor)
        self.assertTrue(all(lr(k) >= lr(k + 997) for k in range(warm, n, 50021)))
        self.assertEqual(ovl.learning_rate({"phase": "conversation", "global_step": 5}), float(s["conversation"]))


class CoverageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyGPT(vocab_size=16, embed_dim=8, num_heads=2, num_layers=1, max_seq_len=4,
                             dropout=0, attn_impl="manual")
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def batch(self, ids):
        ids = torch.tensor([ids])
        return {"inputs": torch.ones_like(ids), "targets": torch.ones_like(ids),
                "mask": ids >= 0, "target_ids": ids}

    def ctl(self, cursor, exclude=None):
        c = {"phase": "wikipedia", "global_step": 0, "cursor": cursor}
        if exclude is not None:
            c["exclude"] = {"wikipedia": exclude}
        return c

    def test_contiguous_batch_advances_cursor(self):
        _, cursor, _ = ovl.step(self.model, self.opt, self.batch([10, 11, 12, 13]), self.ctl(10))
        self.assertEqual(cursor, 14)

    def test_gap_without_fork_fails(self):
        with self.assertRaisesRegex(SystemExit, "coverage error"):
            ovl.step(self.model, self.opt, self.batch([10, 11, 15, 16]), self.ctl(10))

    def test_repeat_and_reorder_fail(self):
        for ids in ([10, 10, 11, 12], [11, 10, 12, 13], [9, 10, 11, 12]):
            with self.assertRaisesRegex(SystemExit, "coverage error"):
                ovl.step(self.model, self.opt, self.batch(ids), self.ctl(10))

    def test_gap_allowed_exactly_at_excluded_range(self):
        _, cursor, _ = ovl.step(self.model, self.opt, self.batch([10, 11, 15, 16]), self.ctl(10, [[12, 3]]))
        self.assertEqual(cursor, 17)

    def test_excluded_range_at_cursor_is_skipped(self):
        _, cursor, _ = ovl.step(self.model, self.opt, self.batch([13, 14, 15, 16]), self.ctl(10, [[10, 3]]))
        self.assertEqual(cursor, 17)

    def test_gap_of_wrong_size_or_place_fails(self):
        for ids, exclude in (([10, 11, 16, 17], [[12, 3]]),   # gap larger than the excluded range
                             ([10, 11, 14, 15], [[12, 3]]),   # gap smaller than the excluded range
                             ([10, 13, 14, 15], [[12, 3]])):  # gap not where the range is
            with self.assertRaisesRegex(SystemExit, "coverage error"):
                ovl.step(self.model, self.opt, self.batch(ids), self.ctl(10, exclude))

    def test_exclusion_only_applies_to_its_phase(self):
        c = {"phase": "conversation", "global_step": 0, "cursor": 10, "exclude": {"wikipedia": [[12, 3]]}}
        with self.assertRaisesRegex(SystemExit, "coverage error"):
            ovl.step(self.model, self.opt, self.batch([10, 11, 15, 16]), c)


class WeightsMatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.states = [tiny_state(i) for i in range(4)]
        write_chain(self.run, fake_chain(list(zip(("initial", "progress", "base", "final"), self.states, strict=True))))

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, sd):
        path = self.run / "weights.pt"
        torch.save(sd, path)
        a = argparse.Namespace(run=str(self.run), data=None, weights=str(path), segment=None, random=0, all=False, anchor=None,
                               no_anchor=True)
        ovl.cmd_verify(a)
        return json.loads((self.run / "verifications.jsonl").read_text().splitlines()[-1])

    def test_final_and_base_weights_match_their_entries(self):
        self.assertEqual(self.verify(self.states[3])["checks"][0]["chain_entry"], 3)
        self.assertEqual(self.verify(self.states[2])["checks"][0]["chain_entry"], 2)

    def test_progress_weights_are_not_a_released_model(self):
        with self.assertRaises(SystemExit):
            self.verify(self.states[1])

    def test_unknown_or_modified_weights_fail(self):
        sd = {k: v.clone() for k, v in self.states[3].items()}
        sd["b"][0] += 1e-6
        with self.assertRaises(SystemExit):
            self.verify(sd)
        report = json.loads((self.run / "verifications.jsonl").read_text().splitlines()[-1])
        self.assertEqual(report["result"], "FAIL")
        self.assertIsNone(report["checks"][0]["chain_entry"])


if __name__ == "__main__":
    unittest.main()
