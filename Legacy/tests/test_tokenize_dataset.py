import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from openverifiablellm.manifest_chain import compute_manifest_hash
from openverifiablellm.tokenizer import (
    BPETokenizer,
    SentencePieceTokenizer,
    create_tokenizer,
    hash_tokenizer_config,
    load_tokenizer,
    tokenize_dataset,
    train_tokenizer,
    verify_tokenized_dataset,
)
from openverifiablellm.utils import (
    compute_merkle_root,
    compute_sha256,
    generate_merkle_proof,
    verify_merkle_proof,
)
from openverifiablellm.verify import CheckStatus

SAMPLE_CORPUS = """Artificial intelligence and verifiable computing are foundational to modern AI systems.
Reproducibility guarantees that computation produces identical outputs across different execution environments.
Deterministic dataset tokenization bridges preprocessed text and model training.
Cryptographic Merkle trees ensure that no chunk of training data can be silently altered or tampered with.
"""


class TestTokenizerContract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.text_file = self.tmp / "sample.txt"
        self.text_file.write_text(SAMPLE_CORPUS, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bpe_train_load_encode_decode(self):
        tok_dir = self.tmp / "bpe_model"
        tok = BPETokenizer(vocab_size=100, min_frequency=1)

        # Before training or loading, encode/decode must fail
        with self.assertRaises(RuntimeError):
            tok.encode("hello")
        with self.assertRaises(RuntimeError):
            tok.decode([1, 2])

        tok.train(self.text_file, tok_dir)
        self.assertTrue((tok_dir / "vocab.json").is_file())
        self.assertTrue((tok_dir / "merges.txt").is_file())

        ids = tok.encode("verifiable computing")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        decoded = tok.decode(ids)
        self.assertIn("verifiable", decoded)

        # Test loading from directory into a fresh instance
        new_tok = BPETokenizer()
        new_tok.load(tok_dir)
        new_ids = new_tok.encode("verifiable computing")
        self.assertEqual(ids, new_ids)
        self.assertEqual(new_tok.decode(new_ids), decoded)

    def test_sentencepiece_train_load_encode_decode(self):
        tok_dir = self.tmp / "spm_model"
        tok = SentencePieceTokenizer(vocab_size=100)

        with self.assertRaises(RuntimeError):
            tok.encode("hello")
        with self.assertRaises(RuntimeError):
            tok.decode([1, 2])

        tok.train(self.text_file, tok_dir)
        self.assertTrue((tok_dir / "spm.model").is_file())
        self.assertTrue((tok_dir / "spm.vocab").is_file())

        ids = tok.encode("verifiable computing")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        decoded = tok.decode(ids)
        self.assertIn("verifiable", decoded)

        # Test loading from directory into fresh instance
        new_tok = SentencePieceTokenizer()
        new_tok.load(tok_dir)
        new_ids = new_tok.encode("verifiable computing")
        self.assertEqual(ids, new_ids)
        self.assertEqual(new_tok.decode(new_ids), decoded)

    def test_load_tokenizer_auto_detection(self):
        bpe_dir = self.tmp / "bpe_tok"
        spm_dir = self.tmp / "spm_tok"

        train_tokenizer(self.text_file, save_path=bpe_dir, tokenizer_type="bpe", vocab_size=100, min_frequency=1)
        train_tokenizer(self.text_file, save_path=spm_dir, tokenizer_type="sentencepiece", vocab_size=100)

        loaded_bpe = load_tokenizer(bpe_dir)
        self.assertIsInstance(loaded_bpe, BPETokenizer)

        loaded_spm = load_tokenizer(spm_dir)
        self.assertIsInstance(loaded_spm, SentencePieceTokenizer)

        with self.assertRaises(NotADirectoryError):
            load_tokenizer(self.tmp / "nonexistent")

        empty_dir = self.tmp / "empty_dir"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_tokenizer(empty_dir)


class TestTokenizeDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.text_file = self.tmp / "input.txt"
        self.text_file.write_text(SAMPLE_CORPUS, encoding="utf-8")
        self.tok_dir = self.tmp / "tokenizer"
        train_tokenizer(self.text_file, save_path=self.tok_dir, tokenizer_type="bpe", vocab_size=120, min_frequency=1)
        self.output_bin = self.tmp / "output.bin"
        self.manifest_file = self.tmp / "tokenized_manifest.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_streaming_tokenization_and_manifest(self):
        manifest = tokenize_dataset(
            input_file=self.text_file,
            tokenizer=self.tok_dir,
            output_file=self.output_bin,
            manifest_path=self.manifest_file,
            dtype="uint32",
        )

        self.assertTrue(self.output_bin.is_file())
        self.assertTrue(self.manifest_file.is_file())

        self.assertGreater(manifest["total_tokens"], 0)
        self.assertEqual(manifest["total_bytes"], manifest["total_tokens"] * 4)
        self.assertEqual(self.output_bin.stat().st_size, manifest["total_bytes"])

        # Validate token array content
        tokens = np.fromfile(self.output_bin, dtype="<u4")
        self.assertEqual(len(tokens), manifest["total_tokens"])

        # Validate manifest content
        self.assertEqual(manifest["tokenized_dataset_sha256"], compute_sha256(file_path=self.output_bin))
        self.assertEqual(manifest["input_dataset_sha256"], compute_sha256(file_path=self.text_file))
        self.assertEqual(manifest["merkle_root"], compute_merkle_root(self.output_bin, chunk_size=manifest["chunk_size_bytes"]))

    def test_uint16_dtype_support(self):
        out_bin = self.tmp / "out_u16.bin"
        manifest = tokenize_dataset(
            input_file=self.text_file,
            tokenizer=self.tok_dir,
            output_file=out_bin,
            dtype="uint16",
        )
        self.assertEqual(manifest["dtype"], "uint16")
        self.assertEqual(manifest["total_bytes"], manifest["total_tokens"] * 2)
        tokens = np.fromfile(out_bin, dtype="<u2")
        self.assertEqual(len(tokens), manifest["total_tokens"])

    def test_invalid_dtype_and_arguments(self):
        with self.assertRaises(ValueError):
            tokenize_dataset(self.text_file, self.tok_dir, self.output_bin, dtype="float32")

        with self.assertRaises(ValueError):
            tokenize_dataset(self.text_file, self.tok_dir, self.output_bin, chunk_size_bytes=0)

        with self.assertRaises(FileNotFoundError):
            tokenize_dataset(self.tmp / "nonexistent.txt", self.tok_dir, self.output_bin)

        with self.assertRaises(TypeError):
            tokenize_dataset(self.text_file, object(), self.output_bin)

    def test_samefile_rejection(self):
        with self.assertRaises(ValueError):
            tokenize_dataset(self.text_file, self.tok_dir, self.text_file)

    def test_manifest_samefile_rejection(self):
        # Manifest pointing to input file directly
        with self.assertRaises(ValueError):
            tokenize_dataset(
                self.text_file,
                self.tok_dir,
                self.output_bin,
                manifest_path=self.text_file,
                write_manifest=True,
            )

        # Manifest pointing to output file
        with self.assertRaises(ValueError):
            tokenize_dataset(
                self.text_file,
                self.tok_dir,
                self.output_bin,
                manifest_path=self.output_bin,
                write_manifest=True,
            )

        # Manifest pointing to symlink of input file
        symlink_in = self.tmp / "input_link.txt"
        symlink_in.symlink_to(self.text_file)
        with self.assertRaises(ValueError):
            tokenize_dataset(
                self.text_file,
                self.tok_dir,
                self.output_bin,
                manifest_path=symlink_in,
                write_manifest=True,
            )

    def test_atomic_output_symlink_replacement(self):
        # Ensure that if output_bin is an existing symlink pointing elsewhere,
        # it is replaced without mutating the original symlink target
        decoy_target = self.tmp / "decoy.txt"
        decoy_target.write_bytes(b"untouched original content")
        self.output_bin.symlink_to(decoy_target)

        tokenize_dataset(
            input_file=self.text_file,
            tokenizer=self.tok_dir,
            output_file=self.output_bin,
            write_manifest=False,
        )

        self.assertFalse(self.output_bin.is_symlink())
        self.assertTrue(self.output_bin.is_file())
        self.assertEqual(decoy_target.read_bytes(), b"untouched original content")


    def test_token_id_range_validation(self):
        class DummyNegativeTokenizer:
            def encode(self, text):
                return [-1, 5]

        class DummyOverflowTokenizer:
            def encode(self, text):
                return [70000]

        with self.assertRaises(ValueError):
            tokenize_dataset(self.text_file, DummyNegativeTokenizer(), self.output_bin, write_manifest=False)

        with self.assertRaises(ValueError):
            tokenize_dataset(self.text_file, DummyOverflowTokenizer(), self.output_bin, dtype="uint16", write_manifest=False)

    def test_manifest_provenance_requirement(self):
        class DummyValidTokenizer:
            def encode(self, text):
                return [1, 2, 3]

        # Fails when write_manifest=True without valid tokenizer dir
        with self.assertRaises(ValueError):
            tokenize_dataset(self.text_file, DummyValidTokenizer(), self.output_bin, write_manifest=True)

        # Passes cleanly when write_manifest=False
        m = tokenize_dataset(self.text_file, DummyValidTokenizer(), self.output_bin, write_manifest=False)
        self.assertGreater(m["total_tokens"], 0)

    def test_determinism_across_runs(self):
        out1 = self.tmp / "run1.bin"
        out2 = self.tmp / "run2.bin"

        m1 = tokenize_dataset(self.text_file, self.tok_dir, out1)
        m2 = tokenize_dataset(self.text_file, self.tok_dir, out2)

        self.assertEqual(out1.read_bytes(), out2.read_bytes())
        self.assertEqual(m1["tokenized_dataset_sha256"], m2["tokenized_dataset_sha256"])
        self.assertEqual(m1["merkle_root"], m2["merkle_root"])
        self.assertEqual(m1["total_tokens"], m2["total_tokens"])

    def test_merkle_chunk_proof_verification(self):
        # Create a larger file with smaller chunks so we have multiple Merkle leaves
        large_text = self.tmp / "large.txt"
        large_text.write_text(SAMPLE_CORPUS * 50, encoding="utf-8")
        out_bin = self.tmp / "multi_chunk.bin"

        chunk_size = 64  # Small chunk size in bytes to create multiple leaves
        manifest = tokenize_dataset(
            input_file=large_text,
            tokenizer=self.tok_dir,
            output_file=out_bin,
            chunk_size_bytes=chunk_size,
        )

        merkle_root = manifest["merkle_root"]
        file_bytes = out_bin.read_bytes()
        total_chunks = (len(file_bytes) + chunk_size - 1) // chunk_size

        self.assertGreater(total_chunks, 1)

        # Generate and verify proof for chunk 0
        chunk_0_bytes = file_bytes[:chunk_size]
        proof_0 = generate_merkle_proof(out_bin, chunk_index=0, chunk_size=chunk_size)
        self.assertTrue(verify_merkle_proof(chunk_0_bytes, proof_0, merkle_root))

        # Generate and verify proof for chunk 1
        chunk_1_bytes = file_bytes[chunk_size : chunk_size * 2]
        proof_1 = generate_merkle_proof(out_bin, chunk_index=1, chunk_size=chunk_size)
        self.assertTrue(verify_merkle_proof(chunk_1_bytes, proof_1, merkle_root))

        # Mutated chunk data must fail verification
        tampered_bytes = bytearray(chunk_0_bytes)
        tampered_bytes[0] ^= 0xFF
        self.assertFalse(verify_merkle_proof(bytes(tampered_bytes), proof_0, merkle_root))


class TestTokenizedVerificationReport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.text_file = self.tmp / "input.txt"
        self.text_file.write_text(SAMPLE_CORPUS, encoding="utf-8")
        self.tok_dir = self.tmp / "tokenizer"
        train_tokenizer(self.text_file, save_path=self.tok_dir, tokenizer_type="bpe", vocab_size=120, min_frequency=1)

        # Create a mock preprocessing manifest to test chain linking
        self.prev_manifest = self.tmp / "dataset_manifest.json"
        prev_data = {
            "version": "1.0.0",
            "processed_sha256": compute_sha256(file_path=self.text_file),
            "step": "preprocessing",
        }
        self.prev_manifest.write_text(json.dumps(prev_data, sort_keys=True), encoding="utf-8")

        self.out_bin = self.tmp / "dataset.bin"
        self.manifest_file = self.tmp / "tokenized_manifest.json"

        tokenize_dataset(
            input_file=self.text_file,
            tokenizer=self.tok_dir,
            output_file=self.out_bin,
            manifest_path=self.manifest_file,
            previous_manifest_path=self.prev_manifest,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_verification_report_all_pass(self):
        report = verify_tokenized_dataset(
            tokenized_file=self.out_bin,
            manifest_path=self.manifest_file,
            tokenizer_path=self.tok_dir,
            input_file=self.text_file,
            previous_manifest_path=self.prev_manifest,
        )

        self.assertTrue(report.all_passed)
        self.assertEqual(len(report.failed), 0)
        pass_names = [c.name for c in report.passed]
        self.assertIn("tokenized_file_exists", pass_names)
        self.assertIn("tokenized_sha256", pass_names)
        self.assertIn("merkle_root", pass_names)
        self.assertIn("input_dataset_sha256", pass_names)
        self.assertIn("tokenizer_config", pass_names)
        self.assertIn("parent_manifest_chain", pass_names)

    def test_tampered_binary_detection(self):
        raw_bytes = bytearray(self.out_bin.read_bytes())
        raw_bytes[0] ^= 0xFF
        self.out_bin.write_bytes(bytes(raw_bytes))

        report = verify_tokenized_dataset(
            tokenized_file=self.out_bin,
            manifest_path=self.manifest_file,
        )

        self.assertFalse(report.all_passed)
        failed_names = [c.name for c in report.failed]
        self.assertIn("tokenized_sha256", failed_names)
        self.assertIn("merkle_root", failed_names)

    def test_tampered_input_file_detection(self):
        self.text_file.write_text("Modified text content", encoding="utf-8")

        report = verify_tokenized_dataset(
            tokenized_file=self.out_bin,
            manifest_path=self.manifest_file,
            input_file=self.text_file,
        )

        self.assertFalse(report.all_passed)
        failed_names = [c.name for c in report.failed]
        self.assertIn("input_dataset_sha256", failed_names)

    def test_tampered_previous_manifest_chain_detection(self):
        tampered_prev = {
            "version": "1.0.0",
            "processed_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        }
        self.prev_manifest.write_text(json.dumps(tampered_prev), encoding="utf-8")

        report = verify_tokenized_dataset(
            tokenized_file=self.out_bin,
            manifest_path=self.manifest_file,
            previous_manifest_path=self.prev_manifest,
        )

        self.assertFalse(report.all_passed)
        failed_names = [c.name for c in report.failed]
        self.assertIn("parent_manifest_chain", failed_names)

    def test_missing_manifest_detection(self):
        report = verify_tokenized_dataset(
            tokenized_file=self.out_bin,
            manifest_path=self.tmp / "nonexistent_manifest.json",
        )
        self.assertFalse(report.all_passed)
        self.assertEqual(report.failed[0].name, "manifest_exists")


if __name__ == "__main__":
    unittest.main()
