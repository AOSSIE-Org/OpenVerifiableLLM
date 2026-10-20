import io
import pickle
import sys
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import CIFARDataset, get_dataset  # noqa: E402


class TestCIFARDatasetIntegrity(unittest.TestCase):
    def test_cifar_offline_fallback_on_network_error(self):
        """Simulate an offline environment where CIFAR download fails.
        
        Should cleanly fall back to fixed synthetic dataset and clean up partial archives.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("dataset._download", side_effect=urllib.error.URLError("Network unreachable")):
                dataset = CIFARDataset(data_dir=tmp_path)
                self.assertEqual(dataset._images.shape, (2048, 3, 32, 32))
                self.assertEqual(dataset._labels.shape, (2048,))
                self.assertEqual(dataset.encoded.shape, (2048,))
                # Ensure no partial tar.gz artifact remained
                self.assertFalse((tmp_path / "cifar-10-python.tar.gz").exists())

    def test_cifar_raises_on_corrupt_tar_archive(self):
        """Pre-existing corrupted archive must raise, never fall back to synthetic."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corrupt_tgz = tmp_path / "cifar-10-python.tar.gz"
            corrupt_tgz.write_bytes(b"not a valid tar.gz file content")

            with self.assertRaises((tarfile.TarError, EOFError, OSError)):
                CIFARDataset(data_dir=tmp_path)

    def test_cifar_raises_on_unsafe_tar_traversal(self):
        """Tar archives containing path traversal members must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tgz_path = tmp_path / "cifar-10-python.tar.gz"

            with tarfile.open(tgz_path, "w:gz") as tar:
                payload = b"dummy"
                info = tarfile.TarInfo(name="../escaped_file.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            with self.assertRaises(ValueError) as ctx:
                CIFARDataset(data_dir=tmp_path)
            self.assertIn("Unsafe path in archive", str(ctx.exception))

    def test_cifar_raises_on_corrupted_pickle_batch(self):
        """Corrupted local batch files must raise UnpicklingError or EOFError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            batch_dir = tmp_path / "cifar-10-batches-py"
            batch_dir.mkdir(parents=True)

            # Write corrupted garbage to data_batch_1
            (batch_dir / "data_batch_1").write_bytes(b"garbage-non-pickle-data")

            with self.assertRaises((pickle.UnpicklingError, EOFError)):
                CIFARDataset(data_dir=tmp_path)

    def test_cifar_raises_on_malformed_batch_structure(self):
        """Valid pickle but invalid internal CIFAR dictionary structure must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            batch_dir = tmp_path / "cifar-10-batches-py"
            batch_dir.mkdir(parents=True)

            # Valid pickle, but missing expected b"data" and b"labels" keys
            malformed_dict = {b"wrong_key": [1, 2, 3]}
            with open(batch_dir / "data_batch_1", "wb") as f:
                pickle.dump(malformed_dict, f)

            with self.assertRaises(ValueError) as ctx:
                CIFARDataset(data_dir=tmp_path)
            self.assertIn("Corrupted or invalid CIFAR batch format", str(ctx.exception))

    def test_cifar_raises_on_missing_batch_file(self):
        """If batch directory exists but a required batch is missing, must raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            batch_dir = tmp_path / "cifar-10-batches-py"
            batch_dir.mkdir(parents=True)

            # Only write batch 1, batches 2..5 missing
            valid_batch = {b"data": [[0] * 3072], b"labels": [1]}
            with open(batch_dir / "data_batch_1", "wb") as f:
                pickle.dump(valid_batch, f)

            with self.assertRaises(FileNotFoundError):
                CIFARDataset(data_dir=tmp_path)

    def test_cifar_loads_valid_batches_correctly(self):
        """When valid batch files 1..5 exist, load and concatenate tensors correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            batch_dir = tmp_path / "cifar-10-batches-py"
            batch_dir.mkdir(parents=True)

            # Write valid batches 1 through 5 with 4 samples each
            for i in range(1, 6):
                sample_data = [[float(i * 10)] * 3072 for _ in range(4)]
                sample_labels = [i] * 4
                with open(batch_dir / f"data_batch_{i}", "wb") as f:
                    pickle.dump({b"data": sample_data, b"labels": sample_labels}, f)

            dataset = CIFARDataset(data_dir=tmp_path)
            self.assertEqual(dataset._images.shape, (20, 3, 32, 32))
            self.assertEqual(dataset._labels.shape, (20,))
            self.assertEqual(dataset.encoded.shape, (20,))

            # Test get_batch sampling
            xb, yb = dataset.get_batch(batch_size=8, device="cpu")
            self.assertEqual(xb.shape, (8, 3, 32, 32))
            self.assertEqual(yb.shape, (8,))

    def test_synthetic_generator_does_not_mutate_global_rng(self):
        """Synthetic generator must use dedicated seed and leave global torch RNG untouched."""
        state_before = torch.get_rng_state()
        CIFARDataset._synthetic_dataset(num_samples=16)
        state_after = torch.get_rng_state()
        self.assertTrue(torch.equal(state_before, state_after))

    def test_get_dataset_factory(self):
        """Factory function get_dataset handles 'cifar' with custom data_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("dataset._download", side_effect=urllib.error.URLError("Offline")):
                ds = get_dataset("cifar", data_dir=tmp_path)
                self.assertIsInstance(ds, CIFARDataset)
                self.assertEqual(ds.name, "cifar")
                self.assertEqual(ds.vocab_size, 10)

    def test_cifar_raises_on_filesystem_permission_error(self):
        """Filesystem errors during download must propagate and not trigger synthetic fallback."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("dataset._download", side_effect=PermissionError("Read-only filesystem")):
                with self.assertRaises(PermissionError):
                    CIFARDataset(data_dir=tmp_path)


if __name__ == "__main__":
    unittest.main()
