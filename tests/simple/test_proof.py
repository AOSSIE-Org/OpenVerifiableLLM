"""Unit tests for simple/proof.py.

Verifies article lookup, segment intersection, fork exclusion proofs,
and cryptographic data binding against real schema fixtures matching
release/simple-run/chain.jsonl and published prepared Wikipedia streams.
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

        # Real documents.jsonl schema: identity is [source_sha256, page_id, revision_id]
        self.docs = [
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "12",
                    "1372193622",
                ],
                "offset": 0,
                "target_start": 0,
                "targets": 9037,
                "tokens": 9037,
            },
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "39",
                    "1368916291",
                ],
                "offset": 9037,
                "target_start": 9037,
                "targets": 5721,
                "tokens": 5721,
            },
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "9999",
                    "1371999999",
                ],
                "offset": 14758,
                "target_start": 274560000,
                "targets": 2000,
                "tokens": 2000,
            },
        ]
        doc_lines = [json.dumps(d) for d in self.docs]
        doc_content = "\n".join(doc_lines) + "\n"
        doc_path = self.wiki_dir / "documents.jsonl"
        doc_path.write_text(doc_content, encoding="utf-8")
        doc_hash = hashlib.sha256(doc_path.read_bytes()).hexdigest()

        # Real stream.json schema with files table containing bytes, path, and sha256
        stream_data = {
            "documents": len(self.docs),
            "files": [
                {
                    "bytes": len(doc_content.encode("utf-8")),
                    "path": "documents.jsonl",
                    "sha256": doc_hash,
                }
            ],
            "phase": "wikipedia",
            "schema": "ovl.stream.v1",
            "targets": 7118121139,
            "tokens": 7118121139,
        }
        stream_bytes = json.dumps(stream_data).encode("utf-8")
        (self.wiki_dir / "stream.json").write_bytes(stream_bytes)
        self.stream_hash = hashlib.sha256(stream_bytes).hexdigest()

        # Real chain entries copied from release/simple-run/chain.jsonl cursors
        self.chain = [
            {
                "control": {
                    "batches": "10ea13010a13668c8bf9e2884e16bdbd16f8933d9bf4d09eda0c47459133eddb",
                    "cursor": 0,
                    "epoch": 0,
                    "global_step": 0,
                    "phase": "wikipedia",
                    "phase_step": 0,
                    "pos": [0, 0],
                },
                "data": {
                    "conversation": "e179de793decc82e8cecfbfabd7d449d1caf145a988c4c881a9a041dcda36b7f",
                    "wikipedia": self.stream_hash,
                },
                "environment": {
                    "cublas_workspace": ":4096:8",
                    "cuda": "12.8",
                    "cudnn": 91002,
                    "gpu": "NVIDIA GeForce RTX 5090",
                    "torch": "2.8.0+cu128",
                },
                "index": 0,
                "kind": "initial",
                "model": "2b167bcb407f94ac3f1dead8f316ea180823a1cb40489b8a1ff974bf2bcd959d",
                "previous": None,
                "recipe": "d0d9fc61c46b0f0e0b63e336579e7443860141452d881b1e7297ad46439f9fc5",
                "state": "64041a4736bd76b64b149c2a784950a312fb19ed641c7eb87e7300c171d002fc",
                "time": 1792422250.861772,
            },
            {
                "control": {
                    "batches": "e3f19a5269c4824fb1615d433b678fa54a6fc23ea99fd82c3e83f3f400f66809",
                    "cursor": 274561003,
                    "epoch": 0,
                    "global_step": 18080,
                    "phase": "wikipedia",
                    "phase_step": 18080,
                    "pos": [14363834, 3072],
                },
                "index": 1,
                "kind": "progress",
                "model": "c32f02fac9e87d0c4279aa39a0fa06b5cc06d5b141d5ff29503cb8431c4ca180",
                "previous": None,
                "state": "ef5090325fcf3c392503406ac2f7e6bd99118d267c38fbe4b624df41b48eca8f",
                "time": 1792424215.6430938,
            },
            {
                "control": {
                    "batches": "d5b199348b8d97e5a30ee2ea1a9bc465fd5dc0220a1a4297b4fa0716abdc84dc",
                    "cursor": 547313827,
                    "epoch": 0,
                    "global_step": 36160,
                    "phase": "wikipedia",
                    "phase_step": 36160,
                    "pos": [28727668, 3072],
                },
                "index": 2,
                "kind": "progress",
                "model": "f626d56d900693a129d2b2fe8e9ee20268ec3540a08e16999787ff9e28f11075",
                "previous": None,
                "state": "85efce6bb412cf30a2f1abef232ecdf6cead51d7c040d9b50db12fc4591ee899",
                "time": 1792426180.123456,
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

    def test_find_article_by_page_id(self):
        doc = mp.find_article(self.data_dir, "12")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["identity"][1], "12")
        self.assertEqual(doc["target_start"], 0)

    def test_find_article_by_int_page_id(self):
        doc = mp.find_article(self.data_dir, 12)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["identity"][1], "12")

    def test_find_article_not_found(self):
        doc = mp.find_article(self.data_dir, "99999999")
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
                    "cursor": 274561003,
                    "exclude": {"wikipedia": [[274560000, 2000]]},
                },
                "state": "state_fork",
                "previous": "prev_hash",
            },
        ]
        doc = mp.find_article(self.data_dir, "9999")
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
                    "cursor": 274561003,
                    "exclude": {"wikipedia": [[274560000, 2000]]},
                },
                "state": "state_fork",
                "previous": "prev_hash_1",
            },
            {
                "index": 2,
                "kind": "progress",
                "control": {
                    "phase": "wikipedia",
                    "cursor": 547313827,
                    "exclude": {"wikipedia": [[274560000, 2000]]},
                },
                "state": "state_progress",
                "previous": "prev_hash_2",
            },
        ]
        doc = mp.find_article(self.data_dir, "9999")
        proof = mp.build_membership_proof(doc, fork_chain, True, self.stream_hash)

        self.assertEqual(proof["membership_status"], "excluded")
        self.assertEqual(proof["segments"], [])
        self.assertEqual(proof["segment_details"], [])
        self.assertIn("exclusion_proof", proof)
        # Boundary should point to the initial fork entry (1), not the later progress entry (2)
        self.assertEqual(proof["exclusion_proof"]["at_boundary"], 1)
        self.assertEqual(
            proof["exclusion_proof"]["excluded_range"], [274560000, 274562000]
        )

    def test_membership_trained_single_segment(self):
        doc = mp.find_article(self.data_dir, "12")
        proof = mp.build_membership_proof(doc, self.chain, True, self.stream_hash)
        self.assertEqual(proof["membership_status"], "trained")
        self.assertEqual(proof["segments"], [0])
        self.assertEqual(
            proof["segment_details"][0]["article_overlap_targets"], [0, 9037]
        )
        self.assertEqual(proof["article"]["page_id"], "12")
        self.assertNotIn("exclusion_proof", proof)

    def test_membership_trained_cross_segment(self):
        doc = mp.find_article(self.data_dir, "9999")
        proof = mp.build_membership_proof(doc, self.chain, True, self.stream_hash)
        self.assertEqual(proof["membership_status"], "trained")
        self.assertEqual(proof["segments"], [0, 1])
        self.assertEqual(
            [s["article_overlap_targets"] for s in proof["segment_details"]],
            [[274560000, 274561003], [274561003, 274562000]],
        )
        self.assertNotIn("exclusion_proof", proof)

    def test_membership_unreached_boundary_touch(self):
        doc = {
            "identity": [
                "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                "8888",
                "1371888888",
            ],
            "target_start": 600000000,
            "targets": 1000,
        }
        proof = mp.build_membership_proof(doc, self.chain, True, self.stream_hash)
        self.assertEqual(proof["membership_status"], "unreached")
        self.assertEqual(proof["segments"], [])


if __name__ == "__main__":
    unittest.main()
