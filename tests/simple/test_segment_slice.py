"""Tests for CPU-based prepared segment slice verification."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "simple" / "segment_slice.py"
PYTHON = sys.executable

DOCUMENT_COUNT = 33
TOKENS_PER_DOCUMENT = 2


class SegmentSliceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        (self.data / "source").mkdir()

        self._build_fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def _build_fixture(self):
        """Build a small valid 30 -> 31 Wikipedia segment."""

        docs = b"".join(
            (
                json.dumps(
                    {
                        "tokens": 2,
                        "targets": 2,
                        "offset": i * 2,
                        "target_start": i * 2,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            for i in range(DOCUMENT_COUNT)
        )

        (self.data / "segment-30-31-documents.jsonl").write_bytes(docs)

        # Each document has one normal token followed by EOS.
        tokens = np.tile(
            np.array([11, 2], dtype="<u2"),
            DOCUMENT_COUNT,
        )
        tokens.tofile(self.data / "segment-30-31-tokens.u16")

        mask = np.ones(DOCUMENT_COUNT * 2, dtype=np.uint8)
        mask.tofile(self.data / "segment-30-31-mask.u8")

        # Independently calculate the expected hash for each batch.
        initial_batches = '0' * 64
        running_batches = initial_batches

        for offset in range(0, DOCUMENT_COUNT, 32):
            batch_size = min(32, DOCUMENT_COUNT - offset)

            inputs = np.zeros((batch_size, 512), dtype=np.int64)
            inputs[:, 0] = 1  # BOS
            inputs[:, 1] = 11

            batch_hash = hashlib.sha256(inputs.tobytes(order='C')).digest()
            running_batches = hashlib.sha256(
                bytes.fromhex(running_batches) + batch_hash
            ).hexdigest()

        final_batches = running_batches

        start_control = {
            "phase": "wikipedia",
            "pos": [0, 0],
            "cursor": 0,
            "batches": initial_batches,
            "global_step": 0,
        }

        end_control = {
            "phase": "wikipedia",
            "pos": [len(docs), 0],
            "cursor": DOCUMENT_COUNT * TOKENS_PER_DOCUMENT,
            "batches": final_batches,
            "global_step": 2,
        }

        stream = {"files": [{"bytes": 1_000_000}]}
        stream_bytes = json.dumps(stream).encode("utf-8")
        stream_path = self.data / "source" / "wikipedia-stream.json"
        stream_path.write_bytes(stream_bytes)

        chain = []

        for i in range(32):
            previous = (
                hashlib.sha256(
                    json.dumps(chain[-1], sort_keys=True).encode()
                ).hexdigest()
                if chain
                else None
            )

            entry = {
                "index": i,
                "previous": previous,
            }

            if i == 0:
                entry["data"] = {
                    "wikipedia": hashlib.sha256(stream_bytes).hexdigest(),
                }

            if i == 30:
                entry["control"] = start_control
            elif i == 31:
                entry["control"] = end_control
            else:
                entry["kind"] = "test"

            chain.append(entry)

        chain_path = self.data / "chain.jsonl"
        chain_path.write_text(
            "".join(json.dumps(entry) + "\n" for entry in chain),
            encoding="utf-8",
        )


    def _run(self):
        return subprocess.run(
            [
                PYTHON,
                str(SCRIPT),
                "--start",
                "30",
                "--end",
                "31",
                "--data",
                str(self.data),
            ],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            cwd=ROOT,
        )

    def test_valid_segment_passes(self):
        result = self._run()

        self.assertEqual(
            result.returncode,
            0,
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertIn("REAL SEGMENT CPU REPLAY: PASS", result.stdout)
        self.assertIn("Batches reproduced: 2", result.stdout)

    def test_manifest_tampering_fails(self):
        path = self.data / "source" / "wikipedia-stream.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")

        try:
            result = self._run()
        finally:
            path.write_bytes(original)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest hash does not match", result.stderr)

    def test_token_corruption_fails(self):
        path = self.data / "segment-30-31-tokens.u16"
        original = path.read_bytes()

        corrupted = bytearray(original)
        corrupted[0] ^= 1
        path.write_bytes(corrupted)

        try:
            result = self._run()
        finally:
            path.write_bytes(original)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ROLLING BATCH HASH DOES NOT MATCH", result.stderr)

    def test_mask_corruption_fails(self):
        path = self.data / "segment-30-31-mask.u8"
        original = path.read_bytes()

        corrupted = bytearray(original)
        corrupted[0] = 2
        path.write_bytes(corrupted)

        try:
            result = self._run()
        finally:
            path.write_bytes(original)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-binary mask", result.stderr)

    def test_chain_tampering_fails(self):
        path = self.data / "chain.jsonl"
        original = path.read_bytes()
        lines = original.splitlines()

        entry = json.loads(lines[30])
        entry["control"]["cursor"] += 1
        lines[30] = json.dumps(entry).encode()
        path.write_bytes(b"\n".join(lines) + b"\n")

        try:
            result = self._run()
        finally:
            path.write_bytes(original)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Chain link broken at entry 31", result.stderr)

    def test_missing_token_file_fails(self):
        path = self.data / "segment-30-31-tokens.u16"
        path.unlink()

        result = self._run()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required file", result.stderr)

    def test_invalid_boundary_fails(self):
        result = subprocess.run(
            [
                PYTHON,
                str(SCRIPT),
                "--start",
                "30",
                "--end",
                "32",
                "--data",
                str(self.data),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must immediately follow", result.stderr)


if __name__ == "__main__":
    unittest.main()
