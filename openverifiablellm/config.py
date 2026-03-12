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

    Returns:
        Optional[List[str]]: A list of benchmark identifiers (e.g., "gsm8k", "cais/mmlu", or local file paths)
        when the YAML is present and valid, or None when the file is missing, unreadable, or does not contain
        a "benchmarks" list.
    """
    if not yaml_path.is_file():
        return None

    try:
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict) and "benchmarks" in data:
            raw_benchmarks = data["benchmarks"]
            if isinstance(raw_benchmarks, list):
                benchmarks = [b.strip() for b in raw_benchmarks if isinstance(b, str) and b.strip()]
                if benchmarks:
                    return benchmarks
        logger.warning(
            "benchmarks.yaml found but does not contain a valid "
            "non-empty 'benchmarks' list; falling back to defaults."
        )
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read/parse %s: %s", yaml_path, exc)
    return None

def load_config(cli_benchmarks: Optional[str] = None) -> BenchmarkConfig:
    """Build a :class:`BenchmarkConfig` by resolving benchmark sources."""
    config = BenchmarkConfig()

    # Priority 1: CLI flag
    if cli_benchmarks:
        benchmarks = [b.strip() for b in cli_benchmarks.split(",") if b.strip()]
        if benchmarks:
            config.benchmarks = benchmarks
            logger.info("Benchmarks from CLI: %s", config.benchmarks)
            return config
        else:
            logger.warning("CLI provided empty benchmarks; falling back to next priority.")

    # Priority 2: YAML config file
    yaml_benchmarks = _load_from_yaml(Path(BENCHMARKS_YAML))
    if yaml_benchmarks:
        config.benchmarks = yaml_benchmarks
        logger.info("Benchmarks from %s: %s", BENCHMARKS_YAML, config.benchmarks)
        return config

    # Priority 3: defaults
    logger.info("Using default benchmarks: %s", config.benchmarks)
    return config