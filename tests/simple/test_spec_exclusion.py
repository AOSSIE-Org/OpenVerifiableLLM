"""`next_target` against the three properties proved in spec/OVL/Exclusion.lean.

The Lean proofs are about a copy of `next_target`; this test ties that copy to the code that ran.
For ranges sorted by start (what `excluded` returns), `next_target(c, ranges)`:
never moves back, skips only excluded targets, and returns a target that is not excluded.
"""
import importlib.util
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simple")]
_spec = importlib.util.spec_from_file_location("ovl_simple_spec", ROOT / "simple" / "ovl.py")
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)


def in_some(ranges, t):
    return any(s <= t < s + n for s, n in ranges)


class NextTargetMatchesSpec(unittest.TestCase):
    def test_random_sorted_ranges(self):
        rng = random.Random(0)
        for _ in range(5000):
            raw = [[rng.randrange(60), rng.randrange(12)] for _ in range(rng.randrange(6))]
            ranges = ovl.excluded({"phase": "wikipedia", "exclude": {"wikipedia": raw}})
            c = rng.randrange(80)
            r = ovl.next_target(c, ranges)
            self.assertGreaterEqual(r, c)                                         # le_nextTarget
            self.assertTrue(all(in_some(ranges, t) for t in range(c, r)))         # skipped_excluded
            self.assertFalse(in_some(ranges, r), (c, ranges, r))                  # nextTarget_not_excluded

    def test_unsorted_ranges_can_fail(self):
        # Why the spec needs sorted ranges: a later range can push the cursor into an earlier one.
        ranges = [(20, 5), (10, 12)]  # 10 -> 22, inside (20, 5)
        self.assertTrue(in_some(ranges, ovl.next_target(10, ranges)))


if __name__ == "__main__":
    unittest.main()
