import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifacts import (  # noqa: E402
    build_merkle_manifest,
    compute_sha256,
    generate_merkle_proof,
    merkle_root_from_leaf_hashes,
    verify_merkle_proof,
)


class ArtifactMerkleTests(unittest.TestCase):
    def test_merkle_manifest_matches_manual_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"abcdefghij")

            manifest = build_merkle_manifest(path, chunk_size=4)
            leaves = [
                compute_sha256(data=b"abcd"),
                compute_sha256(data=b"efgh"),
                compute_sha256(data=b"ij"),
            ]

            self.assertEqual(manifest["chunk_count"], 3)
            self.assertEqual(manifest["merkle_root"], merkle_root_from_leaf_hashes(leaves))
            self.assertEqual(manifest["sha256"], compute_sha256(file_path=path))

    def test_merkle_proof_verifies_chunk_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"abcdefghij")
            manifest = build_merkle_manifest(path, chunk_size=4)
            proof = generate_merkle_proof(path, 1, chunk_size=4)

            self.assertTrue(verify_merkle_proof(b"efgh", proof, manifest["merkle_root"]))
            self.assertFalse(verify_merkle_proof(b"EFGH", proof, manifest["merkle_root"]))

    def test_empty_file_root_is_empty_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.bin"
            path.write_bytes(b"")
            manifest = build_merkle_manifest(path, chunk_size=4)

            self.assertEqual(manifest["chunk_count"], 0)
            self.assertEqual(manifest["merkle_root"], compute_sha256(data=b""))


if __name__ == "__main__":
    unittest.main()
