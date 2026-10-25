import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from verifier import find_weights  # noqa: E402


class FindWeightsPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "model.safetensors").write_bytes(b"w")
        self.outside = root / "secret.bin"
        self.outside.write_bytes(b"s")

    def tearDown(self):
        self._tmp.cleanup()

    def test_normal_name_is_accepted(self):
        path = find_weights(self.model_dir, {"weights": "model.safetensors"})
        self.assertEqual(path, (self.model_dir / "model.safetensors").resolve())

    def test_parent_relative_name_is_rejected(self):
        with self.assertRaises(ValueError):
            find_weights(self.model_dir, {"weights": "../secret.bin"})

    def test_absolute_path_is_rejected(self):
        with self.assertRaises(ValueError):
            find_weights(self.model_dir, {"weights": str(self.outside)})

    def test_checkpoint_artifact_key_is_checked_too(self):
        with self.assertRaises(ValueError):
            find_weights(self.model_dir, {"model_checkpoint_artifact": "../secret.bin"})

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlink_escaping_model_dir_is_rejected(self):
        link = self.model_dir / "link.safetensors"
        link.symlink_to(self.outside)
        with self.assertRaises(ValueError):
            find_weights(self.model_dir, {"weights": "link.safetensors"})


if __name__ == "__main__":
    unittest.main()
