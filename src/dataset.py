"""Datasets for OpenVerifiableLLM.

Replaces the 16-char "abcdabcdabcd" toy with real corpora. Two things matter for
the determinism story:

1. ``get_batch`` draws its indices from the GLOBAL torch RNG (``torch.randint``
   with no explicit generator). That means the batch sequence is captured by the
   same ``torch.get_rng_state()`` the checkpoint already saves/restores, so a
   segmented replay stays bitwise-exact *for free* -- PROVIDED the caller draws
   the batch as the FIRST statement inside the training loop, before any other
   RNG consumer (dropout, etc.). reproducibility.py and run_experiment.py do.

2. Model SIZE (not corpus size) drives the safetensors file size and therefore a
   non-degenerate Merkle tree. Corpus size only affects narrative credibility and
   how long a divergence takes to show up.

Network: on a RunPod pod (unrestricted) the loaders download the real corpora.
Offline (e.g. CI / a locked-down sandbox) the shakespeare loader falls back to a
small bundled public-domain sample so the audit still runs.
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Single reliable single-file sources. wikitext is handled via HF `datasets`.
# Each source includes a pinned SHA-256 hash for integrity verification.
_SOURCES = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "sha256": "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    },
    "enwik8": {
        "url": "https://mattmahoney.net/dc/enwik8.zip",
        "sha256": "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc",
    },
}

_ENWIK8_CHARS = 90_000_000  # standard enwik8 train split is the first 90M bytes


def _download(url, dest, expected_hash=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()

    # Verify hash if provided
    if expected_hash:
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch for {dest.name}: expected {expected_hash}, got {actual_hash}"
            )

    dest.write_bytes(data)
    return dest


def load_corpus(name):
    """Return the raw text for a char-level corpus as a Python str."""
    name = name.lower()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if name == "shakespeare":
        local = DATA_DIR / "shakespeare.txt"
        if not local.exists():
            try:
                print(" ~> downloading tinyshakespeare ...", file=sys.stderr)
                _download(_SOURCES["shakespeare"]["url"], local,
                         expected_hash=_SOURCES["shakespeare"]["sha256"])
            except Exception as exc:  # offline / blocked -> bundled sample
                sample = DATA_DIR / "shakespeare_sample.txt"
                if sample.exists():
                    print(
                        f" ~> download failed ({type(exc).__name__}); using bundled "
                        f"offline sample {sample.name}. Determinism results are valid; "
                        f"for the full 1 MB corpus run on a networked machine.",
                        file=sys.stderr,
                    )
                    return sample.read_text(encoding="utf-8")
                raise
        return local.read_text(encoding="utf-8")

    if name == "enwik8":
        local = DATA_DIR / "enwik8.txt"
        if not local.exists():
            zpath = DATA_DIR / "enwik8.zip"
            if not zpath.exists():
                print(" ~> downloading enwik8 (~36 MB) ...", file=sys.stderr)
                _download(_SOURCES["enwik8"]["url"], zpath,
                         expected_hash=_SOURCES["enwik8"]["sha256"])
            with zipfile.ZipFile(zpath) as zf:
                # Validate all members to prevent path traversal attacks
                for member in zf.namelist():
                    target_path = (DATA_DIR / member).resolve()
                    if not target_path.is_relative_to(DATA_DIR.resolve()):
                        raise ValueError(f"Unsafe path in archive: {member}")
                raw = zf.read("enwik8")[:_ENWIK8_CHARS]
            local.write_bytes(raw)
        return local.read_text(encoding="latin-1")

    if name == "wikitext":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "wikitext requires the HuggingFace `datasets` package: "
                "`pip install datasets`. (shakespeare and enwik8 need no extra deps.)"
            ) from exc
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return "\n".join(ds["text"])

    raise ValueError(f"unknown text corpus {name!r}")


class CharDataset:
    """Character-level corpus with replay-exact batch sampling."""

    def __init__(self, name="shakespeare", block_size=128):
        self.name = name
        self.block_size = block_size

        text = load_corpus(name)
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        # Back-compat alias: eval.py / global_manifest.py hash dataset.encoded.
        self.encoded = self.data

    def get_batch(self, batch_size=16, block_size=None, device="cpu"):
        """Sample a batch. MUST be the first RNG draw in the training loop.

        Uses the global torch RNG (no explicit generator) so the batch sequence
        is part of the state captured by torch.get_rng_state().
        """
        block_size = block_size or self.block_size
        # y is data[i+1 : i+1+block_size], so the last valid start i satisfies
        # i + block_size + 1 <= len(data)  ->  max_start = len(data) - block_size - 1.
        max_start = self.data.size(0) - block_size - 1
        if max_start < 0:
            raise ValueError(
                f"corpus too short ({self.data.size(0)} tokens) for block_size={block_size}"
            )
        ix = torch.randint(0, max_start + 1, (batch_size,))  # upper bound exclusive
        x = torch.stack([self.data[i : i + block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)


class CIFARDataset:
    """CIFAR-10 for the CNN row (stretch / different modality).

    Downloads the official python pickle on a networked machine; offline it
    synthesizes a fixed random tensor dataset with a DEDICATED generator (so the
    global RNG -- and therefore training determinism -- is untouched). The
    synthetic path exercises the conv code path; real accuracy needs the download.
    """

    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    def __init__(self, image_size=32):
        self.name = "cifar"
        self.vocab_size = 10  # num classes; named vocab_size for a uniform API
        self.num_classes = 10
        self.image_size = image_size
        self._images, self._labels = self._load()
        self.encoded = self._labels  # for manifest hashing

    def _load(self):
        local_dir = DATA_DIR / "cifar-10-batches-py"
        try:
            if not local_dir.exists():
                import tarfile

                tgz = DATA_DIR / "cifar-10-python.tar.gz"
                if not tgz.exists():
                    print(" ~> downloading CIFAR-10 (~170 MB) ...", file=sys.stderr)
                    _download(self.URL, tgz)
                with tarfile.open(tgz) as tf:
                    # Validate all members to prevent path traversal attacks
                    for member in tf.getmembers():
                        target_path = (DATA_DIR / member.name).resolve()
                        if not target_path.is_relative_to(DATA_DIR.resolve()):
                            raise ValueError(f"Unsafe path in archive: {member.name}")
                    tf.extractall(DATA_DIR)
            import pickle

            xs, ys = [], []
            for i in range(1, 6):
                with open(local_dir / f"data_batch_{i}", "rb") as f:
                    d = pickle.load(f, encoding="bytes")
                xs.append(torch.tensor(d[b"data"], dtype=torch.float32))
                ys.extend(d[b"labels"])
            x = torch.cat(xs).view(-1, 3, 32, 32) / 255.0
            y = torch.tensor(ys, dtype=torch.long)
            return x, y
        except Exception as exc:
            print(
                f" ~> CIFAR download/parse failed ({type(exc).__name__}); using a fixed "
                f"synthetic image set (conv path only).",
                file=sys.stderr,
            )
            g = torch.Generator().manual_seed(0)  # dedicated -> global RNG untouched
            x = torch.randn(2048, 3, 32, 32, generator=g)
            y = torch.randint(0, 10, (2048,), generator=g)
            return x, y

    def get_batch(self, batch_size=64, block_size=None, device="cpu"):
        n = self._images.size(0)
        ix = torch.randint(0, n, (batch_size,))  # global RNG -> replay-exact
        return self._images[ix].to(device), self._labels[ix].to(device)


def get_dataset(name, block_size=128):
    """Factory: text corpora -> CharDataset, 'cifar' -> CIFARDataset."""
    if name.lower() == "cifar":
        return CIFARDataset()
    return CharDataset(name=name, block_size=block_size)


# get_batch draws from the GLOBAL torch RNG -> replay-exact when called first in-loop

