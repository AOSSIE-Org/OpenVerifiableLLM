"""Unit tests for near-duplicate and topic-level information leakage analysis."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import near_duplicates as nd  # noqa: E402
import ovl  # noqa: E402
from data import OFFSET  # noqa: E402


class NearDuplicatesSubsequenceTests(unittest.TestCase):
    def test_empty_or_oversized_subsequence(self):
        arr = np.array([1, 2, 3], dtype=np.uint16)
        self.assertEqual(
            nd.find_token_subsequences(arr, np.array([], dtype=np.uint16)), []
        )
        self.assertEqual(
            nd.find_token_subsequences(
                arr, np.array([1, 2, 3, 4], dtype=np.uint16)
            ),
            [],
        )

    def test_single_token_subsequence(self):
        arr = np.array([10, 20, 10, 30, 10], dtype=np.uint16)
        self.assertEqual(
            nd.find_token_subsequences(arr, np.array([10], dtype=np.uint16)),
            [0, 2, 4],
        )

    def test_multi_token_subsequence(self):
        arr = np.array([1, 2, 3, 4, 1, 2, 3, 5, 1, 2], dtype=np.uint16)
        sub = np.array([1, 2, 3], dtype=np.uint16)
        self.assertEqual(nd.find_token_subsequences(arr, sub), [0, 4])

    def test_overlapping_pattern(self):
        arr = np.array([1, 1, 1, 1], dtype=np.uint16)
        sub = np.array([1, 1], dtype=np.uint16)
        self.assertEqual(nd.find_token_subsequences(arr, sub), [0, 1, 2])


class EntityPatternTests(unittest.TestCase):
    def setUp(self):
        vocab = {
            "<pad>": 0,
            "3I/ATLAS": 1,
            " 3I/ATLAS": 2,
            "comet": 3,
            "interstellar": 4,
        }
        self.tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))

    def test_build_entity_patterns_adds_offset(self):
        patterns = nd.build_entity_patterns(self.tok, "3I/ATLAS")
        self.assertTrue(len(patterns) >= 1)
        # All tokens must be shifted by data.OFFSET
        for p in patterns:
            self.assertTrue((p["tokens"] >= OFFSET).all())

    def test_build_entity_patterns_no_hardcoded_aliases(self):
        patterns = nd.build_entity_patterns(self.tok, "3I/ATLAS")
        texts = [p["text"] for p in patterns]
        # Must not contain any hardcoded aliases or unmentioned objects
        self.assertNotIn("3I/Borisov", texts)
        self.assertNotIn("C/2019 Q4", texts)
        self.assertNotIn("comet ATLAS", texts)

    def test_build_entity_patterns_with_explicit_aliases(self):
        patterns = nd.build_entity_patterns(
            self.tok, "3I/ATLAS", extra_aliases=["comet"]
        )
        texts = [p["text"] for p in patterns]
        self.assertTrue(any("comet" in t.lower() for t in texts))


class ScanEntityMentionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wiki_dir = Path(self.tmp.name) / "wikipedia"
        self.wiki_dir.mkdir(parents=True)

        # 3 documents matching real schema: [source_sha256, page_id, revision_id]
        # Doc 1: Page ID 101 (Primary, to be excluded)
        # Doc 2: Page ID 102 (Mentions 3I/ATLAS at token offset 5)
        # Doc 3: Page ID 103 (Control document)
        docs = [
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "101",
                    "1372000001",
                ],
                "offset": 0,
                "tokens": 10,
                "target_start": 0,
                "targets": 10,
            },
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "102",
                    "1372000002",
                ],
                "offset": 10,
                "tokens": 20,
                "target_start": 10,
                "targets": 20,
            },
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "103",
                    "1372000003",
                ],
                "offset": 30,
                "tokens": 15,
                "target_start": 30,
                "targets": 15,
            },
        ]
        with (self.wiki_dir / "documents.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        # Stream arrays: total 45 tokens
        # In Doc 2 (offset 10), insert pattern [99, 100] at offset 15 (relative 5)
        tokens = np.zeros(45, dtype=np.uint16)
        tokens[15:17] = [99, 100]
        tokens.tofile(self.wiki_dir / "tokens.u16")

        mask = np.ones(45, dtype=np.uint8)
        mask.tofile(self.wiki_dir / "mask.u8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_finds_secondary_mentions_and_excludes_primary(self):
        patterns = [
            {"text": "3I/ATLAS", "tokens": np.array([99, 100], dtype=np.uint16)}
        ]
        results = nd.scan_entity_mentions(
            self.tmp.name,
            patterns,
            excluded_pages=["101"],
            ctx=10,
        )
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["page_id"], "102")
        self.assertEqual(res["window_idx"], 0)
        self.assertEqual(res["window_start"], 10)
        self.assertEqual(res["mention_span"], (5, 7))

    def test_scan_finds_mentions_crossing_chunk_boundary(self):
        # Insert a 2-token pattern [99, 100] across boundary: at doc offset 9 (global 19)
        tokens = np.zeros(45, dtype=np.uint16)
        tokens[19:21] = [99, 100]
        tokens.tofile(self.wiki_dir / "tokens.u16")

        patterns = [
            {"text": "3I/ATLAS", "tokens": np.array([99, 100], dtype=np.uint16)}
        ]
        results = nd.scan_entity_mentions(
            self.tmp.name,
            patterns,
            excluded_pages=["101"],
            ctx=10,
        )
        # Should emit both window 0 (idx 9..10) and window 1 (idx 0..1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["window_idx"], 0)
        self.assertEqual(results[0]["mention_span"], (9, 10))
        self.assertEqual(results[1]["window_idx"], 1)
        self.assertEqual(results[1]["mention_span"], (0, 1))

    def test_deduplicate_windows_preserves_multiple_spans(self):
        w1 = {
            "window_start": 10,
            "window_len": 10,
            "mention_span": (2, 4),
            "pattern": "a",
        }
        w2 = {
            "window_start": 10,
            "window_len": 10,
            "mention_span": (6, 8),
            "pattern": "b",
        }
        deduped = nd.deduplicate_windows([w1, w2])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["mention_spans"], [(2, 4), (6, 8)])
        self.assertEqual(deduped[0]["patterns"], ["a", "b"])


class PassageLossAndAttributionTests(unittest.TestCase):
    def setUp(self):
        vocab = {"<pad>": 0, "token": 1, "atlas": 2}
        self.tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))

        # Tiny model recipe on CPU
        recipe = {
            **ovl.RECIPE["model"],
            "embed_dim": 32,
            "num_layers": 1,
            "num_heads": 2,
        }
        self.model_orig = ovl.TinyGPT(**recipe).to("cpu").eval()
        self.model_fork = ovl.TinyGPT(**recipe).to("cpu").eval()

        self.tok_stream = np.full(20, OFFSET + 1, dtype=np.uint16)
        self.tok_stream[5:7] = OFFSET + 2
        self.mask_stream = np.ones(20, dtype=np.uint8)

    def test_eval_passage_loss_returns_token_attributions(self):
        win = {
            "window_start": 0,
            "window_len": 12,
            "window_idx": 0,
            "mention_span": (5, 7),
        }
        res = nd.eval_passage_loss_and_tokens(
            self.model_orig,
            self.model_fork,
            self.tok_stream,
            self.mask_stream,
            win,
            tokenizer=self.tok,
        )
        self.assertIsNotNone(res)
        self.assertIn("window_loss_orig", res)
        self.assertIn("window_loss_fork", res)
        self.assertIn("token_attributions", res)
        self.assertEqual(len(res["token_attributions"]), 12)
        # Check that mention tokens are correctly marked
        mention_toks = [
            t for t in res["token_attributions"] if t["is_mention"]
        ]
        self.assertEqual(len(mention_toks), 2)


class BootstrapSignificanceTests(unittest.TestCase):
    def test_bootstrap_significance_with_clear_signal(self):
        target_deltas = [0.25, 0.30, 0.28, 0.32, 0.27]
        control_deltas = [0.01, 0.02, -0.01, 0.00, 0.03]
        stats = nd.compute_bootstrap_significance(
            target_deltas, control_deltas, n_resamples=500
        )
        self.assertGreater(stats["net_unlearning"], 0.2)
        self.assertTrue(stats["statistically_significant"])
        self.assertLess(stats["p_value"], 0.05)
        self.assertGreater(stats["ci_95"][0], 0.15)

    def test_bootstrap_significance_with_noise_only(self):
        target_deltas = [0.01, -0.01, 0.02, 0.00]
        control_deltas = [0.01, 0.00, -0.01, 0.02]
        stats = nd.compute_bootstrap_significance(
            target_deltas, control_deltas, n_resamples=500
        )
        self.assertFalse(stats["statistically_significant"])

    def test_bootstrap_empty_safety(self):
        stats = nd.compute_bootstrap_significance([], [0.1])
        self.assertEqual(stats, {"error": "no target deltas"})
        stats2 = nd.compute_bootstrap_significance([0.1], [])
        self.assertEqual(stats2, {"error": "no control deltas"})


class TopicFilterSpecTests(unittest.TestCase):
    def test_make_topic_filter_spec_creates_deterministic_sha_and_merges_ranges(
        self,
    ):
        patterns = [{"text": "3I/ATLAS"}]
        # Two overlapping windows [100, 64] and [140, 64] -> merge to [100, 104]
        # And one separate window [500, 128]
        windows = [
            {"target_start": 500, "target_len": 128},
            {"target_start": 100, "target_len": 64},
            {"target_start": 140, "target_len": 64},
        ]
        dummy_tok_hash = "a" * 64
        spec = nd.make_topic_filter_spec(
            "3I/ATLAS",
            patterns,
            windows,
            stream_targets=1000,
            tokenizer_sha256=dummy_tok_hash,
            aliases=["C/2025 N1"],
            primary_documents_excluded=["101"],
        )
        self.assertEqual(spec["filter_name"], "topic_near_duplicate_filter")
        self.assertEqual(spec["excluded_ranges_count"], 2)
        # Ranges must be sorted by start offset and non-overlapping
        self.assertEqual(spec["ranges"], [[100, 104], [500, 128]])
        self.assertEqual(len(spec["filter_code_sha256"]), 64)
        self.assertEqual(spec["tokenizer_sha256"], dummy_tok_hash)
        self.assertEqual(spec["aliases"], ["C/2025 N1"])
        self.assertEqual(spec["primary_documents_excluded"], ["101"])
        # No fake hand checks by default
        self.assertIsNone(spec["sample_miss_rate"])
        self.assertIsNone(spec["sample_file_sha256"])

        # Validation helper passes
        self.assertTrue(
            nd.validate_topic_filter_spec(spec, stream_targets=1000)
        )

    def test_topic_filter_spec_with_sample_check(self):
        patterns = [{"text": "3I/ATLAS"}]
        windows = [{"target_start": 100, "target_len": 50}]
        sample_data = {"checked": 25, "misses": 0, "false_positives": 0}
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(sample_data, f)
            sample_path = f.name
        try:
            expected_sample_sha = hashlib.sha256(
                Path(sample_path).read_bytes()
            ).hexdigest()
            spec = nd.make_topic_filter_spec(
                "3I/ATLAS",
                patterns,
                windows,
                sample_check_path=sample_path,
            )
            self.assertEqual(spec["sample_miss_rate"], sample_data)
            self.assertEqual(spec["sample_file_sha256"], expected_sample_sha)
            self.assertTrue(nd.validate_topic_filter_spec(spec))
        finally:
            Path(sample_path).unlink(missing_ok=True)

    def test_validate_topic_filter_spec_catches_invalid_ranges_and_hashes(self):
        patterns = [{"text": "3I/ATLAS"}]
        windows = [{"target_start": 100, "target_len": 50}]
        spec = nd.make_topic_filter_spec("3I/ATLAS", patterns, windows)

        # Mismatched code hash
        bad_code_spec = dict(spec)
        bad_code_spec["filter_code_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_code_spec)

        # Invalid tokenizer hash
        bad_tok_spec = dict(spec)
        bad_tok_spec["tokenizer_sha256"] = "invalid_short_hash"
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_tok_spec)

        # Invalid sample_miss_rate (not dict or None)
        bad_sample_spec = dict(spec)
        bad_sample_spec["sample_miss_rate"] = 123
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_sample_spec)

        # Non-integer / fractional coordinates
        bad_spec_frac = dict(spec)
        bad_spec_frac["ranges"] = [[100.5, 50]]
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_spec_frac)

        bad_spec_bool = dict(spec)
        bad_spec_bool["ranges"] = [[True, 50]]
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_spec_bool)

        # Unsorted ranges
        bad_spec = dict(spec)
        bad_spec["ranges"] = [[200, 50], [100, 50]]
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_spec)

        # Overlapping ranges
        bad_spec2 = dict(spec)
        bad_spec2["ranges"] = [[100, 50], [120, 50]]
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_spec2)

        # Non-positive count
        bad_spec3 = dict(spec)
        bad_spec3["ranges"] = [[100, 0]]
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_spec3)

        # Exceeds stream targets
        bad_spec4 = dict(spec)
        bad_spec4["ranges"] = [[100, 50]]
        with self.assertRaises(ValueError):
            nd.validate_topic_filter_spec(bad_spec4, stream_targets=120)

    def test_spec_sha256_computes_hex(self):
        patterns = [{"text": "3I/ATLAS"}]
        windows = [{"target_start": 100, "target_len": 50}]
        spec = nd.make_topic_filter_spec("3I/ATLAS", patterns, windows)
        digest = nd.spec_sha256(spec)
        self.assertEqual(len(digest), 64)


class NearDuplicatesCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "data"
        self.orig_dir = self.root / "original"
        self.fork_dir = self.root / "fork"
        self.wiki_dir = self.data_dir / "wikipedia"
        self.tok_dir = self.data_dir / "tokenizer"

        self.wiki_dir.mkdir(parents=True)
        self.tok_dir.mkdir(parents=True)
        self.orig_dir.mkdir(parents=True)
        self.fork_dir.mkdir(parents=True)

        vocab = {
            "<pad>": 0,
            "3I/ATLAS": 1,
            " 3I/ATLAS": 2,
            "comet": 3,
            "interstellar": 4,
            "2I/Borisov": 5,
        }
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
        tokenizer.save(str(self.tok_dir / "tokenizer.json"))

        # Real documents schema
        docs = [
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "101",
                    "1372000001",
                ],
                "offset": 0,
                "tokens": 10,
                "target_start": 0,
                "targets": 10,
            },
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "102",
                    "1372000002",
                ],
                "offset": 10,
                "tokens": 20,
                "target_start": 10,
                "targets": 20,
            },
            {
                "identity": [
                    "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                    "103",
                    "1372000003",
                ],
                "offset": 30,
                "tokens": 20,
                "target_start": 30,
                "targets": 20,
            },
        ]
        with (self.wiki_dir / "documents.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        tokens = np.zeros(50, dtype=np.uint16)
        tokens[15] = 1 + OFFSET
        tokens[35] = 5 + OFFSET
        tokens.tofile(self.wiki_dir / "tokens.u16")

        mask = np.ones(50, dtype=np.uint8)
        mask.tofile(self.wiki_dir / "mask.u8")

        recipe = {
            **ovl.RECIPE,
            "model": {
                **ovl.RECIPE["model"],
                "embed_dim": 32,
                "num_layers": 1,
                "num_heads": 2,
            },
        }
        (self.orig_dir / "recipe.json").write_text(
            json.dumps(recipe), encoding="utf-8"
        )
        (self.fork_dir / "recipe.json").write_text(
            json.dumps(recipe), encoding="utf-8"
        )

        model = ovl.TinyGPT(**recipe["model"])
        torch.save(model.state_dict(), self.orig_dir / "base-model.pt")
        torch.save(model.state_dict(), self.fork_dir / "base-model.pt")

        e0 = {
            "index": 0,
            "kind": "initial",
            "control": {"phase": "wikipedia", "cursor": 0},
            "previous": None,
        }
        h0 = hashlib.sha256(json.dumps(e0, sort_keys=True).encode()).hexdigest()
        e1 = {
            "index": 1,
            "kind": "fork",
            "control": {
                "phase": "wikipedia",
                "cursor": 0,
                "exclude": {"wikipedia": [[0, 10]]},
            },
            "fork": {
                "from_entry": 0,
                "documents": [
                    [
                        "859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116",
                        "101",
                        "1372000001",
                    ]
                ],
            },
            "previous": h0,
        }
        fork_chain = [e0, e1]
        with (self.fork_dir / "chain.jsonl").open("w", encoding="utf-8") as f:
            for e in fork_chain:
                f.write(json.dumps(e) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_end_to_end_with_filter_output(self):
        filter_out = self.root / "topic_filter.json"
        args = [
            "near_duplicates.py",
            "--data",
            str(self.data_dir),
            "--original",
            str(self.orig_dir),
            "--fork",
            str(self.fork_dir),
            "--entity",
            "3I/ATLAS",
            "--alias",
            "comet",
            "--control-entities",
            "2I/Borisov",
            "--out-filter",
            str(filter_out),
        ]
        buf = io.StringIO()
        with patch("sys.argv", args), patch("sys.stdout", buf):
            nd.main()

        out_text = buf.getvalue()
        report = json.loads(out_text)
        self.assertEqual(report["entity"], "3I/ATLAS")
        self.assertEqual(report["aliases"], ["comet"])
        self.assertIn("102", report["articles_with_mentions"])
        self.assertIn("statistics", report)
        self.assertTrue(filter_out.is_file())
        filter_data = json.loads(filter_out.read_text(encoding="utf-8"))
        self.assertEqual(filter_data["target_entity"], "3I/ATLAS")
        self.assertEqual(filter_data["aliases"], ["comet"])
        self.assertEqual(len(filter_data["filter_code_sha256"]), 64)
        self.assertEqual(len(filter_data["tokenizer_sha256"]), 64)
        self.assertIsNone(filter_data["sample_miss_rate"])
        self.assertIsNone(filter_data["sample_file_sha256"])
        self.assertIn("topic_filter_spec_sha256", report)

    def test_cli_end_to_end_with_sample_check(self):
        filter_out = self.root / "topic_filter_sample.json"
        sample_path = self.root / "hand_sample.json"
        sample_content = {
            "checked": 20,
            "misses": 0,
            "false_positives": 0,
        }
        sample_path.write_text(json.dumps(sample_content), encoding="utf-8")
        expected_sample_sha = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()

        args = [
            "near_duplicates.py",
            "--data",
            str(self.data_dir),
            "--original",
            str(self.orig_dir),
            "--fork",
            str(self.fork_dir),
            "--entity",
            "3I/ATLAS",
            "--sample-check",
            str(sample_path),
            "--out-filter",
            str(filter_out),
        ]
        buf = io.StringIO()
        with patch("sys.argv", args), patch("sys.stdout", buf):
            nd.main()

        filter_data = json.loads(filter_out.read_text(encoding="utf-8"))
        self.assertEqual(filter_data["sample_miss_rate"], sample_content)
        self.assertEqual(
            filter_data["sample_file_sha256"], expected_sample_sha
        )


if __name__ == "__main__":
    unittest.main()
