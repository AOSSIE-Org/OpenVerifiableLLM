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
        bound, stream_hash = mp.verify_data_commitment(self.data_dir, self.chain[0])
        self.assertTrue(bound)
        self.assertEqual(stream_hash, self.stream_hash)

    def test_verify_data_commitment_tampered_stream(self):
        (self.wiki_dir / "stream.json").write_bytes(b'{"tampered": true}')
        bound, msg = mp.verify_data_commitment(self.data_dir, self.chain[0])
        self.assertFalse(bound)

    def test_verify_data_commitment_tampered_documents(self):
        with (self.wiki_dir / "documents.jsonl").open("a", encoding="utf-8") as f:
            f.write("{}\n")
        bound, msg = mp.verify_data_commitment(self.data_dir, self.chain[0])
        self.assertFalse(bound)

    def test_verify_data_commitment_missing_wikipedia_hash(self):
        tampered_entry = dict(self.chain[0])
        tampered_entry["data"] = {}
        bound, msg = mp.verify_data_commitment(self.data_dir, tampered_entry)
        self.assertFalse(bound)
        self.assertIn("no wikipedia data commitment", msg)

    def test_verify_data_commitment_not_initial_kind(self):
        tampered_entry = dict(self.chain[0])
        tampered_entry["kind"] = "progress"
        bound, msg = mp.verify_data_commitment(self.data_dir, tampered_entry)
        self.assertFalse(bound)
        self.assertIn("not 'initial'", msg)

    def test_read_chain_broken_link(self):
        lines = (self.run_dir / "chain.jsonl").read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["previous"] = "0" * 64
        lines[1] = json.dumps(entry, sort_keys=True)
        (self.run_dir / "chain.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        with self.assertRaises(SystemExit):
            mp.read_chain(self.run_dir)

    def test_membership_fork_exclusion_fork_entry_last(self):
        # Fork entry is the last entry in the chain
        fork_chain = [
            self.chain[0],
            {
                "index": 1,
                "kind": "fork",
                "control": {
                    "phase": "wikipedia",
                    "cursor": 2000,
                    "exclude": {"wikipedia": [[2500, 500]]},
                },
                "state": "state_fork",
                "previous": "prev_hash",
            },
        ]
        doc = mp.find_article(self.data_dir, "3I/ATLAS")
        proof = mp.build_membership_proof(doc, fork_chain, True, self.stream_hash)

        self.assertEqual(proof["membership_status"], "excluded")
        self.assertEqual(proof["segments"], [])
        self.assertEqual(proof["segment_details"], [])
        self.assertIn("exclusion_proof", proof)
        self.assertEqual(proof["exclusion_proof"]["at_boundary"], 1)

    def test_membership_fork_exclusion_continued_training(self):
        # Fork was created at boundary 1, and subsequent entries retain exclude control
        fork_chain = [
            self.chain[0],
            {
                "index": 1,
                "kind": "fork",
                "control": {
                    "phase": "wikipedia",
                    "cursor": 2000,
                    "exclude": {"wikipedia": [[2500, 500]]},
                },
                "state": "state_fork",
                "previous": "prev_hash_1",
            },
            {
                "index": 2,
                "kind": "progress",
                "control": {
                    "phase": "wikipedia",
                    "cursor": 3000,
                    "exclude": {"wikipedia": [[2500, 500]]},
                },
                "state": "state_progress",
                "previous": "prev_hash_2",
            },
        ]
        doc = mp.find_article(self.data_dir, "3I/ATLAS")
        proof = mp.build_membership_proof(doc, fork_chain, True, self.stream_hash)

        self.assertEqual(proof["membership_status"], "excluded")
        self.assertEqual(proof["segments"], [])
        self.assertEqual(proof["segment_details"], [])
        self.assertIn("exclusion_proof", proof)
        # Boundary should point to the initial fork entry (1), not the later progress entry (2)
        self.assertEqual(proof["exclusion_proof"]["at_boundary"], 1)
        self.assertEqual(proof["exclusion_proof"]["excluded_range"], [2500, 3000])


if __name__ == "__main__":
    unittest.main()
