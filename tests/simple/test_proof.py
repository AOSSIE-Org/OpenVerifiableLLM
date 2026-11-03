"""Unit tests for simple/proof.py.

Verifies article lookup, segment intersection, fork exclusion proofs,
and cryptographic data binding against synthetic test fixtures.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "simple"))
import proof as mp  # noqa: E402


class TestMembershipProof(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)
        self.data_dir = self.root / "data"
        self.run_dir = self.root / "run"
        self.wiki_dir = self.data_dir / "wikipedia"
        self.wiki_dir.mkdir(parents=True)
        self.run_dir.mkdir(parents=True)

        self.docs = [
            {
                "identity": [736, "Albert Einstein"],
                "target_start": 0,
                "targets": 1000,
                "tokens": 1200,
                "offset": 0,
            },
            {
                "identity": [1042, "Quantum mechanics"],
                "target_start": 1000,
                "targets": 1500,
                "tokens": 1700,
                "offset": 1200,
            },
            {
                "identity": [9999, "3I/ATLAS"],
                "target_start": 2500,
                "targets": 500,
                "tokens": 600,
                "offset": 2900,
            },
        ]
        doc_lines = [json.dumps(d) for d in self.docs]
        doc_content = "\n".join(doc_lines) + "\n"
        doc_path = self.wiki_dir / "documents.jsonl"
        doc_path.write_text(doc_content, encoding="utf-8")
        doc_hash = hashlib.sha256(doc_path.read_bytes()).hexdigest()

        stream_data = {
            "phase": "wikipedia",
            "targets": 3000,
            "files": [{"path": "documents.jsonl", "sha256": doc_hash}],
        }
        stream_bytes = json.dumps(stream_data).encode("utf-8")
        (self.wiki_dir / "stream.json").write_bytes(stream_bytes)
        self.stream_hash = hashlib.sha256(stream_bytes).hexdigest()

        self.chain = [
            {
                "index": 0,
                "kind": "initial",
                "control": {"phase": "wikipedia", "cursor": 0},
                "data": {"wikipedia": self.stream_hash},
                "state": "state_hash_0",
                "previous": None,
            },
            {
                "index": 1,
                "kind": "progress",
                "control": {"phase": "wikipedia", "cursor": 2000},
                "state": "state_hash_1",
                "previous": None,
            },
            {
                "index": 2,
                "kind": "base",
                "control": {"phase": "wikipedia", "cursor": 3000},
                "state": "state_hash_2",
                "previous": None,
            },
        ]
        for i in range(1, len(self.chain)):
            self.chain[i]["previous"] = hashlib.sha256(
                json.dumps(self.chain[i - 1], sort_keys=True).encode()
            ).hexdigest()

        (self.run_dir / "chain.jsonl").write_text(
            "\n".join(json.dumps(e, sort_keys=True) for e in self.chain) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_find_article_by_title_and_case(self):
        doc = mp.find_article(self.data_dir, "Albert Einstein")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["identity"][0], 736)

        doc_lower = mp.find_article(self.data_dir, "albert einstein")
        self.assertIsNotNone(doc_lower)
        self.assertEqual(doc_lower["identity"][1], "Albert Einstein")

    def test_find_article_by_page_id(self):
        doc = mp.find_article(self.data_dir, "736")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["identity"][1], "Albert Einstein")

    def test_find_article_not_found(self):
        doc = mp.find_article(self.data_dir, "Nonexistent Article")
        self.assertIsNone(doc)

    def test_verify_data_commitment(self):
        bound, stream_hash = mp.verify_data_commitment(
            self.data_dir, self.chain[0]
        )
        self.assertTrue(bound)
        self.assertEqual(stream_hash, self.stream_hash)

    def test_membership_single_segment(self):
        doc = mp.find_article(self.data_dir, "Albert Einstein")
        proof = mp.build_membership_proof(
            doc, self.chain, True, self.stream_hash
        )

        self.assertEqual(proof["membership_status"], "trained")
        self.assertEqual(proof["segments"], [0])
        self.assertEqual(proof["article"]["targets"]["start"], 0)
        self.assertEqual(proof["article"]["targets"]["count"], 1000)
        self.assertEqual(
            proof["segment_details"][0]["boundary_start_state"], "state_hash_0"
        )
        self.assertEqual(
            proof["segment_details"][0]["boundary_end_state"], "state_hash_1"
        )

    def test_membership_spanning_two_segments(self):
        doc = mp.find_article(self.data_dir, "Quantum mechanics")
        proof = mp.build_membership_proof(
            doc, self.chain, True, self.stream_hash
        )

        self.assertEqual(proof["membership_status"], "trained")
        self.assertEqual(proof["segments"], [0, 1])
        self.assertEqual(
            proof["segment_details"][0]["article_overlap_targets"], [1000, 2000]
        )
        self.assertEqual(
            proof["segment_details"][1]["article_overlap_targets"], [2000, 2500]
        )

    def test_membership_fork_exclusion(self):
        fork_chain = list(self.chain)
        fork_chain[1] = dict(fork_chain[1])
        fork_chain[1]["control"] = {
            "phase": "wikipedia",
            "cursor": 2000,
            "exclude": {"wikipedia": [[2500, 500]]},
        }

        doc = mp.find_article(self.data_dir, "3I/ATLAS")
        proof = mp.build_membership_proof(
            doc, fork_chain, True, self.stream_hash
        )

        self.assertEqual(proof["membership_status"], "excluded")
        self.assertIn("exclusion_proof", proof)
        self.assertEqual(proof["exclusion_proof"]["at_boundary"], 1)
        self.assertEqual(
            proof["exclusion_proof"]["excluded_range"], [2500, 3000]
        )


if __name__ == "__main__":
    unittest.main()
