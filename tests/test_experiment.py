"""Tests beyond the original Merkle suite.

Split by dependency so the security/crypto core runs ANYWHERE (no torch, no GPU),
while the determinism tests self-skip when torch / a CUDA GPU is absent:

  torch-free (run in CI and the sandbox):
    * verify-before-load rejects a tampered file BEFORE deserialization
    * Merkle tree over a multi-MB artifact is non-degenerate (>1 chunk)
  torch (run on any machine with torch, incl. the CPU baseline):
    * MLP control is run-to-run bitwise reproducible
    * num_layers actually changes the model
    * T6 precision: bf16 disagrees with the fp32 reference
    * T8 matrix: a tiny sweep yields a reference + a bit-different cell
  CUDA-only (run on the pod):
    * T5 TF32 silently diverges from fp32 at the same seed
    * T7 determinism-OFF diverges run-to-run (first_divergence_step is set)
    * T9 DDP all-reduce ordering (needs >=2 GPUs)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import signing  # noqa: E402  (torch-free)
from artifacts import build_merkle_manifest  # noqa: E402  (torch-free)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
HAS_2GPU = HAS_CUDA and torch.cuda.device_count() >= 2


# --------------------------------------------------------------------------- #
# Security: verify the signature BEFORE deserializing (torch-free)
# --------------------------------------------------------------------------- #
class SigningBeforeLoadTests(unittest.TestCase):
    def test_tamper_rejected_before_deserialization(self):
        sk, vk = signing.generate_keypair()
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp) / "ckpt.bin"
            art.write_bytes(b"trusted-weights" * 1000)
            signing.sign_file(art, sk)
            self.assertTrue(signing.verify_file(art, verify_key=vk))

            # Attacker mutates the bytes; the detached signature is now stale.
            art.write_bytes(b"evil-payload" * 1000)
            with self.assertRaises(signing.SignatureError):
                signing.verify_file(art, verify_key=vk)

    def test_missing_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp) / "unsigned.bin"
            art.write_bytes(b"x" * 100)
            with self.assertRaises(signing.SignatureError):
                signing.verify_file(art)

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_verified_torch_load_blocks_tampered_pickle(self):
        # A real torch checkpoint: clean load works; a tampered file raises
        # SignatureError (never reaching torch.load).
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.pt"
            signing.signed_torch_save({"w": torch.zeros(3)}, ckpt)
            obj = signing.verified_torch_load(ckpt)
            self.assertIn("w", obj)
            with open(ckpt, "r+b") as f:
                f.seek(64)
                f.write(b"\xff\xff\xff\xff")
            with self.assertRaises(signing.SignatureError):
                signing.verified_torch_load(ckpt)


# --------------------------------------------------------------------------- #
# Merkle non-degeneracy on a large artifact (torch-free)
# --------------------------------------------------------------------------- #
class MerkleNonDegenerateTests(unittest.TestCase):
    def test_multi_megabyte_file_has_many_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "weights.bin"
            big.write_bytes(os.urandom(5 * 1024 * 1024 + 123))  # ~5 MB
            manifest = build_merkle_manifest(big)  # default 1 MB chunks
            self.assertGreater(manifest["chunk_count"], 1)
            self.assertNotEqual(manifest["merkle_root"], manifest["chunks"][0]["sha256"])


# --------------------------------------------------------------------------- #
# Determinism (torch; CPU is fine)
# --------------------------------------------------------------------------- #
SMOKE = dict(total_steps=6, batch_size=4, block_size=48)


@unittest.skipUnless(HAS_TORCH, "torch required")
class DeterminismTests(unittest.TestCase):
    def test_mlp_control_is_run_to_run_reproducible(self):
        from experiment import run_one
        rec = run_one("mlp", "shakespeare", "fp32", True, device="cpu",
                      overrides=SMOKE, quiet=True)
        self.assertTrue(rec["reproducible"])
        self.assertIsNone(rec["first_divergence_step"])

    def test_num_layers_changes_model(self):
        from model import build_model, count_params
        from config import model_config
        small = build_model("gpt10m", 65, model_config("gpt10m", num_layers=2))
        big = build_model("gpt10m", 65, model_config("gpt10m", num_layers=6))
        self.assertLess(count_params(small), count_params(big))

    def test_t6_bf16_disagrees_with_fp32_reference(self):
        from experiment import run_one
        ref = run_one("mlp", "shakespeare", "fp32", True, device="cpu",
                      overrides=SMOKE, quiet=True)
        bf16 = run_one("mlp", "shakespeare", "bf16", True, device="cpu",
                       overrides=SMOKE, quiet=True)
        # bf16 is itself run-to-run reproducible, but its bits differ from fp32.
        self.assertNotEqual(ref["param_sha256"], bf16["param_sha256"])

    def test_t8_matrix_has_reference_and_divergent_cell(self):
        from experiment import run_one
        from sweep import annotate_reference, cond_label
        recs = []
        for prec, det in [("fp32", True), ("bf16", True)]:
            r = run_one("mlp", "shakespeare", prec, det, device="cpu",
                        overrides=SMOKE, quiet=True)
            r["condition"] = cond_label(prec, det)
            recs.append(r)
        annotate_reference(recs)
        self.assertTrue(any(r["is_reference"] for r in recs))
        self.assertTrue(any(r["vs_fp32_bitwise"] is False for r in recs))


# --------------------------------------------------------------------------- #
# CUDA-only (run on the pod) -- the headline failure exhibits
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_CUDA, "CUDA GPU required (run on the pod)")
class CudaFailureTests(unittest.TestCase):
    def test_t5_tf32_silently_diverges_from_fp32(self):
        from experiment import run_one
        ref = run_one("gpt10m", "shakespeare", "fp32", True, device="cuda",
                      overrides=SMOKE, quiet=True)
        tf32 = run_one("gpt10m", "shakespeare", "tf32", True, device="cuda",
                       overrides=SMOKE, quiet=True)
        self.assertNotEqual(ref["param_sha256"], tf32["param_sha256"])

    def test_t7_determinism_off_diverges_run_to_run(self):
        from experiment import run_one
        rec = run_one("gpt10m", "shakespeare", "fp32", False, device="cuda",
                      overrides=SMOKE, track_full=True, quiet=True)
        # With determinism OFF, embedding/scatter atomics should break run-to-run.
        self.assertFalse(rec["reproducible"])
        self.assertIsNotNone(rec["first_divergence_step"])


@unittest.skipUnless(HAS_2GPU, "needs >=2 CUDA GPUs (run on a multi-GPU pod)")
class DDPTests(unittest.TestCase):
    def test_t9_ddp_scaffold_runs(self):
        # The DDP experiment lives in src/ddp_repro.py; here we only assert it is
        # importable so the scaffold doesn't bit-rot. The real multi-GPU run is
        # launched via torchrun (see RUNBOOK.md).
        import importlib
        self.assertTrue(importlib.import_module("ddp_repro"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
