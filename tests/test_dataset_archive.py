import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataset  # noqa: E402


def _tar(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)


class CIFARArchiveTest(unittest.TestCase):
    def test_symlink_member_is_rejected_and_nothing_is_written_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir, outside = root / "data", root / "outside"
            data_dir.mkdir()
            outside.mkdir()
            link = tarfile.TarInfo("cifar-10-batches-py")
            link.type, link.linkname = tarfile.SYMTYPE, str(outside)
            payload = b"overwritten"
            through = tarfile.TarInfo("cifar-10-batches-py/data_batch_1")
            through.size = len(payload)
            _tar(data_dir / "cifar-10-python.tar.gz", [(link, None), (through, payload)])
            with mock.patch.object(dataset, "DATA_DIR", data_dir):
                ds = dataset.CIFARDataset()
            self.assertEqual(list(outside.iterdir()), [])
            self.assertFalse((data_dir / "cifar-10-batches-py").exists())
            self.assertEqual(tuple(ds._images.shape), (2048, 3, 32, 32))  # synthetic fallback

    def test_regular_archive_still_extracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            info = tarfile.TarInfo("cifar-10-batches-py/readme.html")
            info.size = 2
            _tar(data_dir / "cifar-10-python.tar.gz", [(info, b"ok")])
            with mock.patch.object(dataset, "DATA_DIR", data_dir):
                dataset.CIFARDataset()  # batches missing -> falls back after extracting
            self.assertEqual((data_dir / "cifar-10-batches-py/readme.html").read_bytes(), b"ok")


if __name__ == "__main__":
    unittest.main()
