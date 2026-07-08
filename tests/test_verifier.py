import json
import os
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

from publish import build_model_card, prepare_publish_dir, publish_huggingface, sign_model_dir  # noqa: E402
from ovllm import main as ovllm_main  # noqa: E402
from verifier import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    check_sigstore_bundle,
    resolve_model_reference,
    verify_model_reference,
)


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

    def test_sign_dry_run_is_read_only_and_ignores_signature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            before = (model_dir / "ovllm_manifest.json").read_text()
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
            self.assertIn("--ignore-paths", command)
            self.assertIn(str((model_dir / "model.sig").resolve()), command)
            self.assertEqual((model_dir / "ovllm_manifest.json").read_text(), before)

    def test_sigstore_verify_ignores_signature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            signature = model_dir / "model.sig"
            signature.write_bytes(b"fake-bundle")
            manifest = json.loads((model_dir / "ovllm_manifest.json").read_text())
            manifest["sigstore_identity"] = "person@example.com"
            manifest["sigstore_identity_provider"] = "https://accounts.example.com"

            with patch("verifier.importlib.util.find_spec") as mock_find_spec, \
                 patch("verifier.subprocess.run") as mock_run:
                mock_find_spec.return_value = "mock_spec"
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "ok"
                mock_run.return_value.stderr = ""
                result = check_sigstore_bundle(model_dir, manifest)

            self.assertEqual(result.status, PASS)
            command = mock_run.call_args.args[0]
            self.assertIn("--ignore-paths", command)
            self.assertIn(str(signature.resolve()), command)

    def test_sign_rejects_signature_path_outside_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            outside = Path(tmp) / "outside.sig"
            with self.assertRaises(ValueError):
                sign_model_dir(str(model_dir), signature=str(outside), dry_run=True)

    def test_local_signing_fails_without_github_actions_or_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            with patch.dict("os.environ", {}, clear=True):
                with patch("sys.stderr") as mock_stderr:
                    code = sign_model_dir(str(model_dir), dry_run=False)
                    self.assertEqual(code, 1)

    def test_local_signing_allows_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._prepared_model_dir(tmp)
            with patch("publish.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                with patch.dict("os.environ", {"OVLLM_ALLOW_LOCAL_SIGNING": "true"}):
                    code = sign_model_dir(str(model_dir), dry_run=False)
                    self.assertEqual(code, 0)


class CliErrorHandlingTests(unittest.TestCase):
    def test_verify_missing_manifest_returns_red_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("builtins.print"):
                code = ovllm_main(["verify", tmp])
        self.assertEqual(code, 1)


class PublishTests(unittest.TestCase):
    def test_model_card_has_hf_yaml_metadata(self):
        card = build_model_card(
            "smoke",
            {
                "weights": "model.safetensors",
                "sha256": "abc",
                "merkle_root": "def",
                "chunk_size_bytes": 1024,
                "chunk_count": 1,
            },
        )

        self.assertTrue(card.startswith("---\n"))
        self.assertIn("openverifiablellm", card)

    def test_publish_huggingface_disables_xet_by_default(self):
        class FakeApi:
            def __init__(self, token=None):
                self.token = token

            def create_repo(self, **_kwargs):
                return None

            def upload_folder(self, **_kwargs):
                return None

        class FakeHub:
            HfApi = FakeApi

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                with patch("importlib.import_module", return_value=FakeHub):
                    code = publish_huggingface("Ryoari/openverifiable-smoke", tmp)
                    self.assertEqual(code, 0)
                    self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")


class RemoteResolutionTests(unittest.TestCase):
    def test_remote_refs_default_to_local_ovllm_cache(self):
        class FakeHub:
            @staticmethod
            def snapshot_download(**kwargs):
                self_cache = Path(kwargs["cache_dir"])
                return str(self_cache / "snapshot")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                with patch.dict("sys.modules", {"huggingface_hub": FakeHub}):
                    with patch("verifier.Path.cwd", return_value=Path(tmp)):
                        resolved = resolve_model_reference("Ryoari/openverifiable-smoke")

                self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")

            self.assertEqual(
                resolved,
                Path(tmp) / ".ovllm-cache" / "huggingface" / "snapshot",
            )

    def test_remote_refs_honor_explicit_cache_dir(self):
        class FakeHub:
            @staticmethod
            def snapshot_download(**kwargs):
                return str(Path(kwargs["cache_dir"]) / "snapshot")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("sys.modules", {"huggingface_hub": FakeHub}):
                with patch("verifier.Path.cwd", return_value=Path(tmp)):
                    resolved = resolve_model_reference(
                        "Ryoari/openverifiable-smoke",
                        cache_dir=str(Path(tmp) / "custom-cache"),
                    )

            self.assertEqual(
                resolved,
                Path(tmp) / "custom-cache" / "snapshot",
            )


class ReportFormattingTests(unittest.TestCase):
    def test_text_markdown_html_report_generation(self):
        from verifier import CheckResult, print_report, PASS, FAIL
        
        results = [
            CheckResult("test_pass", PASS, "Pass detail"),
            CheckResult("test_fail", FAIL, "Fail detail", expected="123", actual="456"),
        ]
        
        # Test Markdown formatting
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "report.md"
            print_report(results, format_type="markdown", output_path=str(md_path), ref="dummy-ref")
            md_content = md_path.read_text(encoding="utf-8")
            self.assertIn("# OpenVerifiableLLM Verification Report", md_content)
            self.assertIn("RED", md_content)
            self.assertIn("test_pass", md_content)
            self.assertIn("test_fail", md_content)
            self.assertIn("123", md_content)
            self.assertIn("456", md_content)

        # Test HTML formatting
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "report.html"
            print_report(results, format_type="html", output_path=str(html_path), ref="dummy-ref")
            html_content = html_path.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", html_content)
            self.assertIn("OpenVerifiableLLM Report", html_content)
            self.assertIn("VERDICT: RED", html_content)
            self.assertIn("test_fail", html_content)
            self.assertIn("status-fail", html_content)

        # Test CLI arguments verify execution for formats
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "model.safetensors"
            save_file({"layer.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}, str(weights))
            model_dir = prepare_publish_dir(weights=str(weights), output_dir=str(Path(tmp) / "publish"))
            
            md_path = Path(tmp) / "cli_report.md"
            code = ovllm_main(["verify", str(model_dir), "--allow-unsigned", "--skip-replay", "--format", "markdown", "--output", str(md_path)])
            self.assertEqual(code, 0)
            self.assertTrue(md_path.exists())
            self.assertIn("# OpenVerifiableLLM Verification Report", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
