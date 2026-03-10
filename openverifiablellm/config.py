"""
Configuration loader for benchmark contamination detection.

"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS: List[str] = ["gsm8k", "cais/mmlu"]

BENCHMARKS_YAML = "benchmarks.yaml"


@dataclass
class BenchmarkConfig:
    """Holds all settings needed by the contamination-detection pipeline."""

    benchmarks: List[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARKS))
    n: int = 13
    bloom_capacity: int = 10_000_000
    bloom_error_rate: float = 0.001
    filter_path: Path = Path("data/filter.bin")
    trust_remote_code: bool = False


def _load_from_yaml(yaml_path: Path) -> Optional[List[str]]:
    """
    Attempt to read benchmark identifiers from a YAML file.

    Expected format::

        benchmarks:
          - gsm8k
          - cais/mmlu
          - /path/to/local/dataset.jsonl

    Returns
    """
    if not yaml_path.is_file():
        return None

    try:
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict) and "benchmarks" in data:
            benchmarks = data["benchmarks"]
            if isinstance(benchmarks, list) and all(
                isinstance(b, str) for b in benchmarks
            ):
                return benchmarks
        logger.warning(
            "benchmarks.yaml found but does not contain a valid "
            "'benchmarks' list; falling back to defaults."
        )
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse %s: %s", yaml_path, exc)

    return None


def load_config(cli_benchmarks: Optional[str] = None) -> BenchmarkConfig:
    """Build a :class:`BenchmarkConfig` by resolving benchmark sources."""
    config = BenchmarkConfig()

    # Priority 1: CLI flag
    if cli_benchmarks:
        config.benchmarks = [
            b.strip() for b in cli_benchmarks.split(",") if b.strip()
        ]
        logger.info("Benchmarks from CLI: %s", config.benchmarks)
        return config

    # Priority 2: YAML config file
    yaml_benchmarks = _load_from_yaml(Path(BENCHMARKS_YAML))
    if yaml_benchmarks is not None:
        config.benchmarks = yaml_benchmarks
        logger.info("Benchmarks from %s: %s", BENCHMARKS_YAML, config.benchmarks)
        return config

    # Priority 3: defaults
    logger.info("Using default benchmarks: %s", config.benchmarks)
    return config