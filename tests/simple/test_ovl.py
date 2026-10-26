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
from unittest.mock import MagicMock, patch

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple")]
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

    def test_empty_chain_file_is_reported(self):
        (self.run / "chain.jsonl").write_text("")
        with self.assertRaisesRegex(SystemExit, "chain.jsonl has no chain entries"):
            ovl.read_chain(self.run)

    def test_cut_off_last_line_is_reported_with_its_line_number(self):
        write_chain(self.run, self.chain[:2])
        with (self.run / "chain.jsonl").open("a") as f:
            f.write('{"index": 2, "kind": "ba')
        with self.assertRaisesRegex(SystemExit, r"chain\.jsonl: line 3 is not valid JSON"):
            ovl.read_chain(self.run)

    def test_blank_line_is_reported_with_its_line_number(self):
        write_chain(self.run, self.chain[:2])
        with (self.run / "chain.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(SystemExit, r"chain\.jsonl: line 3 is not valid JSON"):
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

    def test_base_weights_match_although_transition_repeats_them(self):
        # The phase change records the base model's digest again, so the last match is the transition.
        kinds = ("initial", "progress", "base", "transition", "final")
        states = [self.states[0], self.states[1], self.states[2], self.states[2], self.states[3]]
        write_chain(self.run, fake_chain(list(zip(kinds, states, strict=True))))
        report = self.verify(self.states[2])
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["checks"][0]["chain_entry"], 2)

    def test_recipe_mismatch_fails_before_any_replay(self):
        chain = [json.loads(l) for l in (self.run / "chain.jsonl").read_text().splitlines()]
        chain[0]["recipe"] = "0" * 64
        for i in range(1, len(chain)):
            chain[i]["previous"] = hashlib.sha256(json.dumps(chain[i - 1], sort_keys=True).encode()).hexdigest()
        write_chain(self.run, chain)
        a = argparse.Namespace(run=str(self.run), data=str(self.run / "no-data"), weights=None, segment=[0], random=0,
                               all=False, anchor=None, no_anchor=True)
        with self.assertRaisesRegex(SystemExit, "RECIPE differs"):
            ovl.cmd_verify(a)

    def test_matching_recipe_goes_on_to_the_data_check(self):
        chain = [json.loads(l) for l in (self.run / "chain.jsonl").read_text().splitlines()]
        chain[0]["recipe"] = hashlib.sha256(json.dumps(ovl.RECIPE, sort_keys=True).encode()).hexdigest()
        chain[0]["data"] = {"wikipedia": "0" * 64, "conversation": "0" * 64}
        for i in range(1, len(chain)):
            chain[i]["previous"] = hashlib.sha256(json.dumps(chain[i - 1], sort_keys=True).encode()).hexdigest()
        write_chain(self.run, chain)
        for phase in ("wikipedia", "conversation"):
            (self.run / "data" / phase).mkdir(parents=True)
            (self.run / "data" / phase / "stream.json").write_text("{}")
        a = argparse.Namespace(run=str(self.run), data=str(self.run / "data"), weights=None, segment=[0], random=0,
                               all=False, anchor=None, no_anchor=True)
        with self.assertRaisesRegex(SystemExit, "prepared data differs"):  # past the recipe check
            ovl.cmd_verify(a)

    def test_unknown_or_modified_weights_fail(self):
        sd = {k: v.clone() for k, v in self.states[3].items()}
        sd["b"][0] += 1e-6
        with self.assertRaises(SystemExit):
            self.verify(sd)
        report = json.loads((self.run / "verifications.jsonl").read_text().splitlines()[-1])
        self.assertEqual(report["result"], "FAIL")
        self.assertIsNone(report["checks"][0]["chain_entry"])


class NextTargetAndExclusionTests(unittest.TestCase):
    def test_excluded_sorts_and_filters_by_phase(self):
        ctl = {"phase": "wikipedia", "exclude": {"wikipedia": [[50, 10], [10, 5]], "conversation": [[5, 2]]}}
        self.assertEqual(ovl.excluded(ctl), [(10, 5), (50, 10)])

        ctl_conv = {"phase": "conversation", "exclude": {"wikipedia": [[10, 5]]}}
        self.assertEqual(ovl.excluded(ctl_conv), [])

        self.assertEqual(ovl.excluded({"phase": "wikipedia"}), [])

    def test_next_target_empty_ranges(self):
        self.assertEqual(ovl.next_target(10, []), 10)

    def test_next_target_cursor_before_and_after_range(self):
        ranges = [(10, 5)]
        self.assertEqual(ovl.next_target(5, ranges), 5)
        self.assertEqual(ovl.next_target(9, ranges), 9)
        self.assertEqual(ovl.next_target(15, ranges), 15)
        self.assertEqual(ovl.next_target(20, ranges), 20)

    def test_next_target_cursor_inside_range(self):
        ranges = [(10, 5)]
        self.assertEqual(ovl.next_target(11, ranges), 15)
        self.assertEqual(ovl.next_target(14, ranges), 15)

    def test_next_target_boundaries_exact_s_and_exact_end(self):
        ranges = [(10, 5)]
        self.assertEqual(ovl.next_target(10, ranges), 15)
        self.assertEqual(ovl.next_target(15, ranges), 15)

    def test_next_target_back_to_back_contiguous_ranges(self):
        ranges = [(10, 5), (15, 7)]
        self.assertEqual(ovl.next_target(10, ranges), 22)
        self.assertEqual(ovl.next_target(12, ranges), 22)
        self.assertEqual(ovl.next_target(15, ranges), 22)
        self.assertEqual(ovl.next_target(22, ranges), 22)

    def test_next_target_multiple_spaced_ranges(self):
        ranges = [(10, 5), (30, 10)]
        self.assertEqual(ovl.next_target(10, ranges), 15)
        self.assertEqual(ovl.next_target(12, ranges), 15)
        self.assertEqual(ovl.next_target(20, ranges), 20)
        self.assertEqual(ovl.next_target(30, ranges), 40)
        self.assertEqual(ovl.next_target(35, ranges), 40)
        self.assertEqual(ovl.next_target(40, ranges), 40)

    def test_next_target_overlapping_ranges(self):
        ranges = [(10, 8), (14, 6)]
        self.assertEqual(ovl.next_target(10, ranges), 20)
        self.assertEqual(ovl.next_target(15, ranges), 20)


class BenchTests(unittest.TestCase):
    def test_empty_stream_fails(self):
        a = argparse.Namespace(data="dummy", seconds=60, warmup=50)
        with patch.object(ovl, "fresh", return_value=(MagicMock(), MagicMock())), \
             patch.object(ovl, "prefetch", return_value=iter([])), \
             self.assertRaisesRegex(SystemExit, "bench error: data stream yielded 0 batches"):
            ovl.cmd_bench(a)

    def test_insufficient_updates_fails(self):
        batches = [({"inputs": torch.tensor([1])}, None) for _ in range(5)]
        a = argparse.Namespace(data="dummy", seconds=60, warmup=50)
        with patch.object(ovl, "fresh", return_value=(MagicMock(), MagicMock())), \
             patch.object(ovl, "prefetch", return_value=iter(batches)), \
             patch.object(ovl, "step", return_value=(torch.tensor(1.0), 0, None)), \
             self.assertRaisesRegex(SystemExit, "completed 5 updates; at least 51 updates are required"):
            ovl.cmd_bench(a)

    def test_custom_warmup_and_timing(self):
        batches = [({"inputs": torch.tensor([1])}, None) for _ in range(3)]
        a = argparse.Namespace(data="dummy", seconds=60, warmup=2)
        with patch.object(ovl, "fresh", return_value=(MagicMock(), MagicMock())), \
             patch.object(ovl, "prefetch", return_value=iter(batches)), \
             patch.object(ovl, "step", return_value=(torch.tensor(1.5), 0, None)), \
             patch.object(ovl, "environment", return_value={}), \
             patch("builtins.print") as mock_print:
            ovl.cmd_bench(a)
            self.assertTrue(mock_print.called)
            payload = json.loads(mock_print.call_args[0][0])
            self.assertEqual(payload["updates_timed"], 1)
            self.assertEqual(payload["loss"], 1.5)

    def test_warmup_zero_timing(self):
        batches = [({"inputs": torch.tensor([1])}, None) for _ in range(3)]
        a = argparse.Namespace(data="dummy", seconds=60, warmup=0)
        with patch.object(ovl, "fresh", return_value=(MagicMock(), MagicMock())), \
             patch.object(ovl, "prefetch", return_value=iter(batches)), \
             patch.object(ovl, "step", return_value=(torch.tensor(1.5), 0, None)), \
             patch.object(ovl, "environment", return_value={}), \
             patch("builtins.print") as mock_print:
            ovl.cmd_bench(a)
            self.assertTrue(mock_print.called)
            payload = json.loads(mock_print.call_args[0][0])
            self.assertEqual(payload["updates_timed"], 3)
            self.assertEqual(payload["loss"], 1.5)

    def test_negative_warmup_fails_bench(self):
        a = argparse.Namespace(data="dummy", seconds=60, warmup=-1)
        with self.assertRaisesRegex(SystemExit, "bench error: --warmup cannot be negative"):
            ovl.cmd_bench(a)

    def test_cli_negative_warmup_rejected(self):
        with patch.object(sys, "argv", ["ovl.py", "bench", "--data", "dummy", "--warmup", "-5"]), \
             patch("sys.stderr"):
            with self.assertRaises(SystemExit) as cm:
                ovl.main()
            self.assertEqual(cm.exception.code, 2)

    def test_cli_warmup_flag(self):
        with patch.object(sys, "argv", ["ovl.py", "bench", "--data", "dummy", "--warmup", "5"]), \
             patch.object(ovl, "cmd_bench") as mock_bench:
            ovl.main()
            mock_bench.assert_called_once()
            self.assertEqual(mock_bench.call_args[0][0].warmup, 5)

    def test_cli_default_warmup(self):
        with patch.object(sys, "argv", ["ovl.py", "bench", "--data", "dummy"]), \
             patch.object(ovl, "cmd_bench") as mock_bench:
            ovl.main()
            mock_bench.assert_called_once()
            self.assertEqual(mock_bench.call_args[0][0].warmup, 50)


if __name__ == "__main__":
    unittest.main()
