import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    import torch
    from safetensors.torch import save_file

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from publish import prepare_publish_dir, sign_model_dir  # noqa: E402
from verifier import FAIL, PASS, SKIP, verify_model_reference  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch and safetensors required")
class VerifierTests(unittest.TestCase):
    def _prepared_model_dir(self, tmp):
        weights = Path(tmp) / "model.safetensors"
        save_file({"layer.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}, str(weights))
        return prepare_publish_dir(weights=str(weights), output_dir=str(Path(tmp) / "publish"))

    def test_local_verify_allows_unsigned_for_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            results = verify_model_reference(str(model_dir), allow_unsigned=True, skip_replay=True)
            by_name = {r.name: r for r in results}

            self.assertEqual(by_name["artifact_sha256"].status, PASS)
            self.assertEqual(by_name["merkle_root"].status, PASS)
            self.assertEqual(by_name["tensor_sha256"].status, PASS)
            self.assertEqual(by_name["sigstore_bundle"].status, SKIP)

    def test_missing_signature_is_red_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            results = verify_model_reference(str(model_dir), skip_replay=True)
            by_name = {r.name: r for r in results}

            self.assertEqual(by_name["sigstore_bundle"].status, FAIL)

    def test_tampered_weights_fail_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            weights = model_dir / "model.safetensors"
            weights.write_bytes(weights.read_bytes() + b"tamper")

            results = verify_model_reference(str(model_dir), allow_unsigned=True, skip_replay=True)
            by_name = {r.name: r for r in results}

            self.assertEqual(by_name["artifact_sha256"].status, FAIL)
            self.assertEqual(by_name["merkle_root"].status, FAIL)

    def test_sign_dry_run_updates_manifest_before_signing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            with patch("builtins.print") as mock_print:
                code = sign_model_dir(
                    str(model_dir),
                    identity="person@example.com",
                    identity_provider="https://accounts.example.com",
                    use_ambient_credentials=True,
                    dry_run=True,
                )

            self.assertEqual(code, 0)
            command = mock_print.call_args.args[0]
            self.assertIn("--use_ambient_credentials", command)
            self.assertIn(str(model_dir / "model.sig"), command)
            manifest = __import__("json").loads((model_dir / "ovllm_manifest.json").read_text())
            self.assertEqual(manifest["signature"], "model.sig")
            self.assertEqual(manifest["sigstore_identity"], "person@example.com")
            self.assertEqual(manifest["sigstore_identity_provider"], "https://accounts.example.com")
            self.assertIn("sigstore_signed_at", manifest)


if __name__ == "__main__":
    unittest.main()
