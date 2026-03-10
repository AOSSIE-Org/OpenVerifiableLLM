"""
Contamination detection via n-gram matching and Bloom filters.

Provides utilities to:
- Load benchmark datasets (Hugging Face or local files)
- Generate n-grams from text
- Build / load a Bloom filter populated with benchmark n-grams
- Check whether a text chunk is contaminated
"""

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

from rbloom import Bloom

from openverifiablellm.config import BenchmarkConfig

logger = logging.getLogger(__name__)

#  Text fields we look for when extracting questions from benchmarks 
_TEXT_FIELDS = ("question", "prompt", "problem", "input", "text")


def _bloom_hash(item) -> int:
    """Deterministic hash function for rbloom."""
    raw = item.encode("utf-8") if isinstance(item, str) else item
    digest = hashlib.sha256(raw).digest()[:8]
    return int.from_bytes(digest, "little")

#  Benchmark loading                                                  
def fetch_benchmarks(config: BenchmarkConfig) -> List[str]:
    texts: List[str] = []

    for benchmark in config.benchmarks:
        path = Path(benchmark)
        if path.suffix in (".jsonl", ".csv") and path.is_file():
            texts.extend(_load_local(path))
        else:
            texts.extend(_load_hf(benchmark))

    logger.info(
        "Loaded %d benchmark text(s) from %d source(s).",
        len(texts),
        len(config.benchmarks),
    )
    return texts


def _load_hf(dataset_id: str) -> List[str]:
    """Load benchmark texts from a Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to fetch HF benchmarks. "
            "Install it with: pip install datasets"
        ) from exc

    texts: List[str] = []
    try:
        ds = load_dataset(dataset_id, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Could not load HF dataset '%s': %s", dataset_id, exc)
        return texts

    splits = ds.values() if hasattr(ds, "values") else [ds]
    for split in splits:
        for row in split:
            for field in _TEXT_FIELDS:
                if field in row and row[field]:
                    texts.append(str(row[field]))
                    break

    return texts


def _load_local(path: Path) -> List[str]:
    """Load benchmark texts from a local JSONL or CSV file."""
    texts: List[str] = []

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for field in _TEXT_FIELDS:
                    if field in row and row[field]:
                        texts.append(str(row[field]))
                        break
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for field in _TEXT_FIELDS:
                    if field in row and row[field]:
                        texts.append(str(row[field]))
                        break
    else:
        logger.warning("Unsupported local file format: %s", path)

    return texts


#  N-gram utilities

def get_ngrams(text: str, n: int = 13) -> List[str]:
    """
    Generate whitespace-tokenised n-grams from text.

    The text is normalised (lowercased, punctuation stripped) before
    tokenisation.  If the text contains fewer than *n* tokens the
    result is an empty list.

    """
    normalised = _normalise(text)
    tokens = normalised.split()

    if len(tokens) < n:
        return []

    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#  Bloom filter construction

def build_bloom_filter(
    benchmark_texts: List[str],
    config: BenchmarkConfig,
) -> Bloom:
    """
    Build (or load) a Bloom filter populated with benchmark n-grams.

    If ``config.filter_path`` already exists on disk the serialised
    filter is loaded directly.  Otherwise a new filter is created,
    populated, and written to ``config.filter_path``.

    """
    filter_path = Path(config.filter_path)

    # --- load from cache if available ---
    if filter_path.is_file():
        logger.info("Loading existing Bloom filter from %s", filter_path)
        raw = filter_path.read_bytes()
        bloom = Bloom.load_bytes(raw, _bloom_hash)
        return bloom

    logger.info(
        "Building Bloom filter (capacity=%s, error_rate=%s, n=%s) …",
        config.bloom_capacity,
        config.bloom_error_rate,
        config.n,
    )

    bloom = Bloom(config.bloom_capacity, config.bloom_error_rate, hash_func=_bloom_hash)

    ngram_count = 0
    for text in benchmark_texts:
        for ngram in get_ngrams(text, n=config.n):
            bloom.add(ngram)
            ngram_count += 1

    logger.info("Inserted %d n-grams into the Bloom filter.", ngram_count)

    filter_path.parent.mkdir(parents=True, exist_ok=True)
    filter_path.write_bytes(bloom.save_bytes())
    logger.info("Bloom filter saved to %s", filter_path)

    return bloom


#  Contamination checking

def check_contamination(
    text: str,
    bloom_filter: Bloom,
    benchmark_texts: List[str],
    n: int = 13,
) -> bool:
    """
    Determine whether text is contaminated by benchmark data.

    A two-stage process is used:

    1. Bloom filter check - fast probabilistic membership test on
       the n-grams of text.
    2. Exact match verification - if the Bloom filter signals a
       hit, a full substring search against benchmark_texts is
       performed to eliminate false positives.

    """
    ngrams = get_ngrams(text, n=n)
    if not ngrams:
        return False

    normalised_benchmarks: Optional[Set[str]] = None

    for ngram in ngrams:
        if ngram in bloom_filter:
            if normalised_benchmarks is None:
                normalised_benchmarks = {_normalise(bt) for bt in benchmark_texts}

            for nb in normalised_benchmarks:
                if ngram in nb:
                    return True

    return False