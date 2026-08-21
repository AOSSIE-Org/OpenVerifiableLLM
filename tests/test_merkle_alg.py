"""Regression tests for the Merkle algorithm field and legacy compatibility.

Commit 1ec4e5a hardened the Merkle tree with RFC 6962 domain separation but did
not version the manifest format. Every manifest published before it recorded a
root built with the old construction, so the new verifier recomputed a different
root and reported PROVABLY UNTAMPERED artifacts as hash mismatches -- a false
tampering alarm, the worst failure mode for a verification tool.

These tests pin both halves of the fix: manifests are self-describing via
``merkle_alg``, and a manifest without that field is still read as legacy.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    import torch
    from safetensors.torch import save_file

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from artifacts import (  # noqa: E402
    MERKLE_ALG_LEGACY,
    MERKLE_ALG_RFC6962,
    build_merkle_manifest,
    compute_sha256_bytes,
    generate_merkle_proof,
    merkle_root_from_leaf_hashes,
    verify_merkle_proof,
)


class MerkleAlgTests(unittest.TestCase):
    def _leaves(self, n):
        return [compute_sha256_bytes(data=bytes([i])).hex() for i in range(n)]

    def test_legacy_construction_is_bare_concatenation(self):
        """The legacy root must stay reproducible or old manifests break again."""
        leaves = self._leaves(2)
        expected = compute_sha256_bytes(
            data=bytes.fromhex(leaves[0]) + bytes.fromhex(leaves[1])
        ).hex()
        self.assertEqual(
            merkle_root_from_leaf_hashes(leaves, alg=MERKLE_ALG_LEGACY), expected
        )

    def test_rfc6962_domain_separates_leaves_and_nodes(self):
        leaves = self._leaves(2)
        leaf_a = compute_sha256_bytes(data=b"\x00" + bytes.fromhex(leaves[0]))
        leaf_b = compute_sha256_bytes(data=b"\x00" + bytes.fromhex(leaves[1]))
        expected = compute_sha256_bytes(data=b"\x01" + leaf_a + leaf_b).hex()
        self.assertEqual(
            merkle_root_from_leaf_hashes(leaves, alg=MERKLE_ALG_RFC6962), expected
        )

    def test_algorithms_disagree(self):
        leaves = self._leaves(4)
        self.assertNotEqual(
            merkle_root_from_leaf_hashes(leaves, alg=MERKLE_ALG_LEGACY),
            merkle_root_from_leaf_hashes(leaves, alg=MERKLE_ALG_RFC6962),
        )

    def test_legacy_is_second_preimage_vulnerable_and_rfc6962_is_not(self):
        """The security property that motivated the change.

        Under the legacy tree an internal node can be replayed as a leaf: a
        2-leaf tree and a 1-leaf tree whose leaf IS that internal node produce
        the same root. Domain separation breaks the equivalence.
        """
        leaves = self._leaves(2)
        legacy_root = merkle_root_from_leaf_hashes(leaves, alg=MERKLE_ALG_LEGACY)
        forged = merkle_root_from_leaf_hashes([legacy_root], alg=MERKLE_ALG_LEGACY)
        self.assertEqual(legacy_root, forged, "legacy tree should be forgeable")

        hardened = merkle_root_from_leaf_hashes(leaves, alg=MERKLE_ALG_RFC6962)
        forged_hardened = merkle_root_from_leaf_hashes(
            [hardened], alg=MERKLE_ALG_RFC6962
        )
        self.assertNotEqual(
            hardened, forged_hardened, "RFC 6962 tree must resist the replay"
        )

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaises(ValueError):
            merkle_root_from_leaf_hashes(self._leaves(2), alg="md5-vibes")

    def test_manifest_records_its_algorithm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"abcdefgh")
            manifest = build_merkle_manifest(path, chunk_size=4)
            self.assertEqual(manifest["merkle_alg"], MERKLE_ALG_RFC6962)

            legacy = build_merkle_manifest(path, chunk_size=4, alg=MERKLE_ALG_LEGACY)
            self.assertEqual(legacy["merkle_alg"], MERKLE_ALG_LEGACY)
            self.assertNotEqual(manifest["merkle_root"], legacy["merkle_root"])
            # Byte-level facts must not depend on the tree construction.
            self.assertEqual(manifest["sha256"], legacy["sha256"])
            self.assertEqual(manifest["chunk_count"], legacy["chunk_count"])

    def test_proofs_round_trip_under_both_algorithms(self):
        for alg in (MERKLE_ALG_LEGACY, MERKLE_ALG_RFC6962):
            with self.subTest(alg=alg):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "blob.bin"
                    path.write_bytes(b"abcdefghijkl")
                    manifest = build_merkle_manifest(path, chunk_size=4, alg=alg)
                    proof = generate_merkle_proof(path, 1, chunk_size=4, alg=alg)
                    self.assertTrue(
                        verify_merkle_proof(
                            b"efgh", proof, manifest["merkle_root"], alg=alg
                        )
                    )
                    self.assertFalse(
                        verify_merkle_proof(
                            b"EFGH", proof, manifest["merkle_root"], alg=alg
                        )
                    )

    def test_proof_does_not_verify_across_algorithms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"abcdefghijkl")
            manifest = build_merkle_manifest(path, chunk_size=4, alg=MERKLE_ALG_RFC6962)
            proof = generate_merkle_proof(path, 1, chunk_size=4, alg=MERKLE_ALG_RFC6962)
            self.assertFalse(
                verify_merkle_proof(
                    b"efgh", proof, manifest["merkle_root"], alg=MERKLE_ALG_LEGACY
                )
            )


@unittest.skipUnless(HAS_TORCH, "torch and safetensors required")
class VerifierMerkleCompatTests(unittest.TestCase):
    """The published-artifact regression, reproduced end to end."""

    def _model_dir(self, tmp):
        from publish import prepare_publish_dir

        weights = Path(tmp) / "model.safetensors"
        save_file(
            {"layer.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
            str(weights),
        )
        return prepare_publish_dir(weights=str(weights), output_dir=str(Path(tmp) / "pub"))

    def _verify(self, model_dir):
        from verifier import verify_model_reference

        results = verify_model_reference(
            str(model_dir), allow_unsigned=True, skip_replay=True
        )
        return {r.name: r for r in results}

    def _rewrite_manifest(self, model_dir, mutate):
        path = Path(model_dir) / "ovllm_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mutate(manifest)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def test_new_manifests_declare_the_algorithm(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp)
            manifest = json.loads(
                (model_dir / "ovllm_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["merkle_alg"], MERKLE_ALG_RFC6962)
            self.assertEqual(self._verify(model_dir)["merkle_root"].status, "PASS")

    def test_legacy_manifest_without_alg_field_still_verifies(self):
        """A pre-hardening manifest must verify GREEN, not look tampered with."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp)
            weights = model_dir / "model.safetensors"
            legacy_root = build_merkle_manifest(weights, alg=MERKLE_ALG_LEGACY)[
                "merkle_root"
            ]

            def to_legacy(manifest):
                manifest.pop("merkle_alg", None)
                manifest["merkle_root"] = legacy_root

            self._rewrite_manifest(model_dir, to_legacy)

            by_name = self._verify(model_dir)
            self.assertEqual(by_name["merkle_root"].status, "PASS")
            self.assertIn("assumed legacy", by_name["merkle_root"].detail)

    def test_tampering_still_fails_under_legacy_manifest(self):
        """Back-compat must not become a bypass."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp)
            weights = model_dir / "model.safetensors"
            legacy_root = build_merkle_manifest(weights, alg=MERKLE_ALG_LEGACY)[
                "merkle_root"
            ]

            def to_legacy(manifest):
                manifest.pop("merkle_alg", None)
                manifest["merkle_root"] = legacy_root

            self._rewrite_manifest(model_dir, to_legacy)
            weights.write_bytes(weights.read_bytes() + b"tamper")

            by_name = self._verify(model_dir)
            self.assertEqual(by_name["merkle_root"].status, "FAIL")
            self.assertEqual(by_name["artifact_sha256"].status, "FAIL")

    def test_unknown_algorithm_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp)
            self._rewrite_manifest(
                model_dir, lambda m: m.update({"merkle_alg": "md5-vibes"})
            )
            by_name = self._verify(model_dir)
            self.assertEqual(by_name["merkle_root"].status, "FAIL")
            self.assertIn("unknown merkle_alg", by_name["merkle_root"].detail)

    def test_explicit_null_merkle_alg_fails(self):
        """An explicit None/null must not fall back to legacy; only absent key may."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp)
            self._rewrite_manifest(
                model_dir, lambda m: m.update({"merkle_alg": None})
            )
            by_name = self._verify(model_dir)
            self.assertEqual(by_name["merkle_root"].status, "FAIL")
            self.assertIn("unknown merkle_alg", by_name["merkle_root"].detail)


if __name__ == "__main__":
    unittest.main()
