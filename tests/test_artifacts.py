import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifacts import (  # noqa: E402
    _merkle_leaf,
    build_merkle_manifest,
    compute_sha256,
    compute_sha256_bytes,
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


    def test_next_merkle_level_even_and_odd_parity(self):
        from artifacts import _merkle_leaf, _merkle_parent, _next_merkle_level

        h1 = _merkle_leaf(compute_sha256(data=b"1").encode())
        h2 = _merkle_leaf(compute_sha256(data=b"2").encode())
        h3 = _merkle_leaf(compute_sha256(data=b"3").encode())

        # Even level: pairs adjacent nodes
        even_level = [h1, h2]
        even_copy = list(even_level)
        next_even = _next_merkle_level(even_level)
        self.assertEqual(len(next_even), 1)
        self.assertEqual(next_even[0], _merkle_parent(h1, h2))
        self.assertEqual(even_level, even_copy, "Input level list must not be mutated")

        # Odd level: last element is paired with itself without list mutation
        odd_level = [h1, h2, h3]
        odd_copy = list(odd_level)
        next_odd = _next_merkle_level(odd_level)
        self.assertEqual(len(next_odd), 2)
        self.assertEqual(next_odd[0], _merkle_parent(h1, h2))
        self.assertEqual(next_odd[1], _merkle_parent(h3, h3))
        self.assertEqual(odd_level, odd_copy, "Input level list must not be mutated")

    def test_merkle_proof_exhaustive_across_chunk_counts(self):
        """Test proofs for odd and even chunk counts, verifying every chunk and rejecting corruptions."""
        with tempfile.TemporaryDirectory() as tmp:
            test_counts = [1, 2, 3, 4, 5, 7, 8, 9, 13, 16]
            chunk_size = 4
            for count in test_counts:
                path = Path(tmp) / f"artifact_{count}.bin"
                content = b"".join(f"C{i:03d}".encode("ascii") for i in range(count))
                path.write_bytes(content)

                manifest = build_merkle_manifest(path, chunk_size=chunk_size)
                self.assertEqual(manifest["chunk_count"], count)

                for idx in range(count):
                    chunk_bytes = content[idx * chunk_size : (idx + 1) * chunk_size]
                    proof = generate_merkle_proof(path, idx, chunk_size=chunk_size)

                    # Valid chunk verifies against root
                    self.assertTrue(
                        verify_merkle_proof(chunk_bytes, proof, manifest["merkle_root"]),
                        f"Failed for count={count}, chunk={idx}",
                    )

                    # Corrupted chunk rejected
                    corrupt_chunk = chunk_bytes[:-1] + b"X"
                    self.assertFalse(
                        verify_merkle_proof(corrupt_chunk, proof, manifest["merkle_root"])
                    )

                    # Corrupted sibling hash rejected if proof is non-empty
                    if proof:
                        corrupt_proof = [dict(step) for step in proof]
                        original_hex = corrupt_proof[0]["sibling_sha256"]
                        corrupt_hex = ("0" if original_hex[0] != "0" else "1") + original_hex[1:]
                        corrupt_proof[0]["sibling_sha256"] = corrupt_hex
                        self.assertFalse(
                            verify_merkle_proof(chunk_bytes, corrupt_proof, manifest["merkle_root"])
                        )

                        # Corrupted sibling position rejected if sibling is distinct from current node
                        current_leaf_hex = _merkle_leaf(compute_sha256_bytes(data=chunk_bytes)).hex()
                        if original_hex != current_leaf_hex:
                            corrupt_pos_proof = [dict(step) for step in proof]
                            orig_pos = corrupt_pos_proof[0]["sibling_position"]
                            corrupt_pos_proof[0]["sibling_position"] = "left" if orig_pos == "right" else "right"
                            self.assertFalse(
                                verify_merkle_proof(chunk_bytes, corrupt_pos_proof, manifest["merkle_root"])
                            )

    def test_merkle_proof_bounds_and_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_path = Path(tmp) / "empty.bin"
            empty_path.write_bytes(b"")
            with self.assertRaises(ValueError):
                generate_merkle_proof(empty_path, 0, chunk_size=4)

            valid_path = Path(tmp) / "valid.bin"
            valid_path.write_bytes(b"12345678")  # 2 chunks of 4 bytes
            with self.assertRaises(IndexError):
                generate_merkle_proof(valid_path, -1, chunk_size=4)
            with self.assertRaises(IndexError):
                generate_merkle_proof(valid_path, 2, chunk_size=4)

    def test_verify_merkle_proof_malformed_inputs(self):
        root = compute_sha256(data=b"test")
        chunk = b"test"

        # Invalid root hex
        self.assertFalse(verify_merkle_proof(chunk, [], "not_a_hex"))

        # Invalid sibling hex
        self.assertFalse(
            verify_merkle_proof(
                chunk,
                [{"sibling_sha256": "invalid_hex", "sibling_position": "left"}],
                root,
            )
        )

        # Sibling hash wrong length (not 32 bytes)
        self.assertFalse(
            verify_merkle_proof(
                chunk,
                [{"sibling_sha256": "abcd", "sibling_position": "left"}],
                root,
            )
        )

        # Invalid position value
        self.assertFalse(
            verify_merkle_proof(
                chunk,
                [{"sibling_sha256": root, "sibling_position": "diagonal"}],
                root,
            )
        )

        # Missing sibling_position key
        self.assertFalse(
            verify_merkle_proof(
                chunk,
                [{"sibling_sha256": root}],
                root,
            )
        )


if __name__ == "__main__":
    unittest.main()

