import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from verifier import PASS, check_sigstore_bundle  # noqa: E402


class SigstoreBundlePathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "model.sig").write_bytes(b"sig-bytes")
        self.outside = root / "secret.sig"
        self.outside.write_bytes(b"outside-sig-bytes")

    def tearDown(self):
        self._tmp.cleanup()

    def test_normal_in_directory_signature_verifies(self):
        manifest = {
            "signature": "model.sig",
            "sigstore_identity": "signer@example.com",
            "sigstore_identity_provider": "https://accounts.example.com",
        }
        with patch("verifier.importlib.util.find_spec", return_value=True):
            with patch("verifier.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "ok"
                mock_run.return_value.stderr = ""
                result = check_sigstore_bundle(
                    self.model_dir,
                    manifest,
                    expected_identity="signer@example.com",
                    expected_issuer="https://accounts.example.com",
                )

        self.assertEqual(result.status, PASS)
        command = mock_run.call_args.args[0]
        sig_index = command.index("--signature")
        self.assertEqual(command[sig_index + 1], str((self.model_dir / "model.sig").resolve()))

    def test_parent_relative_name_is_rejected(self):
        manifest = {
            "signature": "../secret.sig",
            "sigstore_identity": "signer@example.com",
            "sigstore_identity_provider": "https://accounts.example.com",
        }
        with self.assertRaises(ValueError):
            check_sigstore_bundle(self.model_dir, manifest)

    def test_absolute_path_is_rejected(self):
        manifest = {
            "signature": str(self.outside),
            "sigstore_identity": "signer@example.com",
            "sigstore_identity_provider": "https://accounts.example.com",
        }
        with self.assertRaises(ValueError):
            check_sigstore_bundle(self.model_dir, manifest)

    def test_sigstore_bundle_key_is_checked_too(self):
        manifest = {
            "sigstore_bundle": "../secret.sig",
            "sigstore_identity": "signer@example.com",
            "sigstore_identity_provider": "https://accounts.example.com",
        }
        with self.assertRaises(ValueError):
            check_sigstore_bundle(self.model_dir, manifest)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlink_escaping_model_dir_is_rejected(self):
        link = self.model_dir / "link.sig"
        try:
            link.symlink_to(self.outside)
        except OSError:
            self.skipTest("symlinks require elevated privileges or Developer Mode on Windows")
        manifest = {
            "signature": "link.sig",
            "sigstore_identity": "signer@example.com",
            "sigstore_identity_provider": "https://accounts.example.com",
        }
        with self.assertRaises(ValueError):
            check_sigstore_bundle(self.model_dir, manifest)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_discovered_symlink_escaping_model_dir_is_rejected(self):
        (self.model_dir / "model.sig").unlink()
        link = self.model_dir / "model.sig"
        try:
            link.symlink_to(self.outside)
        except OSError:
            self.skipTest("symlinks require elevated privileges or Developer Mode on Windows")
        manifest = {
            "sigstore_identity": "signer@example.com",
            "sigstore_identity_provider": "https://accounts.example.com",
        }
        with self.assertRaises(ValueError):
            check_sigstore_bundle(self.model_dir, manifest)


if __name__ == "__main__":
    unittest.main()
