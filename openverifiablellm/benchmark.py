"""
benchmark.py
============
Before-vs-After resource consumption analysis.

Compares the **legacy approach** (load everything into memory, hash at once)
against the **new streaming approach** (O(1) memory generator + Incremental
Merkle Tree) on the same Wikipedia dump.

Metrics captured
----------------
* Wall-clock execution time (seconds) via ``time.perf_counter``.
* Peak heap allocation (MB)          via ``tracemalloc``.

Output
------
Prints a GitHub-flavored Markdown table to stdout so the result can be
copy-pasted directly into a Pull Request description.

Usage
-----
    # Download first (≈350 MB):
    # wget https://dumps.wikimedia.org/simplewiki/20260201/simplewiki-20260201-pages-articles-multistream.xml.bz2

    python -m openverifiablellm.benchmark simplewiki-20260201-pages-articles-multistream.xml.bz2

    # Or via the scripts/ helper:
    python scripts/benchmark.py <path>
"""

import argparse
import bz2
import gc
import hashlib
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Optional, Tuple

import defusedxml.ElementTree as ET

from openverifiablellm.incremental_merkle import IncrementalMerkleTree
from openverifiablellm.utils import clean_wikitext, extract_text_from_xml

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for a benchmark result row
# ---------------------------------------------------------------------------
BenchmarkResult = Tuple[
    str,  # approach label
    float,  # wall-clock seconds
    float,  # peak RAM in MB
    Optional[str],  # root hash (hex), or None when no articles were found
    int,  # article count
]


# ---------------------------------------------------------------------------
# Helper: convert tracemalloc peak bytes → MB
# ---------------------------------------------------------------------------
def _bytes_to_mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


# ===========================================================================
# APPROACH 1 — "Old Way" (in-memory)
# ===========================================================================


def _run_old_way(file_path: Path) -> BenchmarkResult:
    """
    Legacy approach: decompress the entire dump, collect ALL article texts
    into a Python list, then build a standard batch Merkle tree from the list.

    Memory profile: O(N) — every article text lives in RAM simultaneously.
    Time profile  : O(N) for loading + O(N log N) for tree construction.
    """
    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    # ----- Step 1: load all texts into memory -----
    all_texts: List[str] = []

    # Detect compression by inspecting the bz2 magic bytes (same logic as
    # extract_text_from_xml) so plain .xml files are also handled correctly.
    with open(file_path, "rb") as _probe:
        _is_bz2 = _probe.read(3) == b"BZh"
    _open_func = bz2.open if _is_bz2 else open

    with _open_func(file_path, "rb") as raw:
        context = ET.iterparse(raw, events=("end",))
        for _event, elem in context:
            if elem.tag.endswith("page"):
                text_elem = elem.find(".//{*}text")
                if text_elem is None:
                    text_elem = elem.find(".//text")
                if text_elem is not None and text_elem.text:
                    cleaned = clean_wikitext(text_elem.text)
                    if cleaned:
                        all_texts.append(cleaned)
                # NOTE: No elem.clear() — intentionally simulating the
                # old code that leaks every parsed element into memory.

    # ----- Step 2: build Merkle tree from the in-memory list -----
    article_count = len(all_texts)

    # Hash each article text to a leaf
    leaves: List[bytes] = [hashlib.sha256(t.encode("utf-8")).digest() for t in all_texts]

    # Batch construction: classic bottom-up Merkle tree
    if not leaves:
        # Surface zero-article runs explicitly rather than producing a
        # spurious matching root hash.
        root_hex: Optional[str] = None
    else:
        current_level = leaves
        while len(current_level) > 1:
            next_level: List[bytes] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(hashlib.sha256(left + right).digest())
            current_level = next_level
        root_hex = current_level[0].hex()

    t_end = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        "Old Way (in-memory)",
        round(t_end - t_start, 3),
        round(_bytes_to_mb(peak_bytes), 2),
        root_hex,
        article_count,
    )


# ===========================================================================
# APPROACH 2 — "New Way" (streaming)
# ===========================================================================


def _run_new_way(file_path: Path) -> BenchmarkResult:
    """
    New streaming approach: yield one article at a time from the generator
    and feed it directly into the IncrementalMerkleTree.

    Memory profile: O(log N) — only the Merkle frontier is kept in RAM.
    Time profile  : O(N log N) — but with vastly lower constant factors
                    because no large list allocation occurs.
    """
    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    tree = IncrementalMerkleTree()
    article_count = 0

    stream = extract_text_from_xml(file_path, stream=True)
    assert stream is not None, "extract_text_from_xml must return a generator when stream=True"
    for article_text in stream:
        tree.append_leaf(article_text)
        article_count += 1

    # Surface zero-article runs as None rather than a spurious sha256(b"") hash.
    root_hex: Optional[str] = tree.get_root_hash() if article_count > 0 else None

    t_end = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        "New Way (streaming)",
        round(t_end - t_start, 3),
        round(_bytes_to_mb(peak_bytes), 2),
        root_hex,
        article_count,
    )


# ===========================================================================
# Reporting: GitHub-Flavored Markdown table
# ===========================================================================


def _render_markdown_table(
    old: BenchmarkResult,
    new: BenchmarkResult,
    file_name: str,
    trials: int,
) -> str:
    """
    Render a GFM Markdown table suitable for direct use in a GitHub PR.

    Calculates speed-up and RAM reduction ratios and appends a legend.
    Results are aggregated medians across multiple alternating-order trials.
    """
    label_old, time_old, ram_old, hash_old, count_old = old
    label_new, time_new, ram_new, hash_new, count_new = new

    # Guard against division by zero on extremely fast runs
    time_ratio = (time_old / time_new) if time_new > 0 else float("inf")
    ram_ratio = (ram_old / ram_new) if ram_new > 0 else float("inf")

    hash_old_str = hash_old if hash_old is not None else "N/A (0 articles)"
    hash_new_str = hash_new if hash_new is not None else "N/A (0 articles)"
    hashes_match = (hash_old == hash_new) and hash_old is not None
    hash_verdict = (
        "YES — identical root hash"
        if hashes_match
        else (
            "NO  — MISMATCH (investigate!)"
            if hash_old is not None
            else "N/A — no articles processed"
        )
    )

    lines = [
        "",
        "## Benchmark Results",
        "",
        f"> **Input file:** `{file_name}`  ",
        f"> **Trials:** {trials} (alternating order, median reported)  ",
        f"> **Articles processed:** old={count_old}, new={count_new}  ",
        f"> **Root hashes match:** {hash_verdict}",
        "",
        "| Metric                        | Old Way (in-memory) | New Way (streaming) | Improvement        |",
        "|-------------------------------|--------------------:|--------------------:|--------------------|",
        f"| Wall-clock time (s)           | `{time_old:>10.3f}` | `{time_new:>10.3f}` | **{time_ratio:,.1f}× faster**    |",
        f"| Peak RAM usage (MB)           | `{ram_old:>10.2f}` | `{ram_new:>10.2f}` | **{ram_ratio:,.1f}× less RAM**   |",
        f"| Root hash                     | `{hash_old_str[:16]}…` | `{hash_new_str[:16]}…` | {'Match' if hashes_match else 'MISMATCH'}              |",
        "",
        "### Notes",
        "- *Peak RAM* is measured with `tracemalloc` (Python heap only; does not include",
        "  OS-level buffers or the bzip2 decompressor's internal state).",
        f"- *Wall-clock time* is the median of {trials} isolated subprocess trials with",
        "  alternating run order to minimise OS pagecache and warm-up bias.",
        "- The Old Way intentionally omits `elem.clear()` to reproduce the OOM behaviour.",
        "- The New Way uses `extract_text_from_xml(..., stream=True)` + `IncrementalMerkleTree` from this PR.",
        "",
    ]
    return "\n".join(lines)


def _render_terminal_table(
    old: BenchmarkResult,
    new: BenchmarkResult,
    file_name: str,
    trials: int,
) -> str:
    """Plain-text box table for terminal output (complements the Markdown table)."""
    label_old, time_old, ram_old, hash_old, count_old = old
    label_new, time_new, ram_new, hash_new, count_new = new
    time_ratio = (time_old / time_new) if time_new > 0 else float("inf")
    ram_ratio = (ram_old / ram_new) if ram_new > 0 else float("inf")
    hashes_match = (hash_old == hash_new) and hash_old is not None

    w = 90
    sep = "─" * w

    def row(col1: str, col2: str, col3: str, col4: str = "") -> str:
        return f"│ {col1:<28} │ {col2:>18} │ {col3:>18} │ {col4:<14} │"

    lines = [
        f"┌{sep}┐",
        f"│{'BEFORE vs. AFTER  —  ' + file_name + f'  ({trials} trials, median)':^{w}}│",
        f"├{sep}┤",
        row("Metric", "Old Way", "New Way", "Improvement"),
        f"├{sep}┤",
        row(
            "Wall-clock time (s)",
            f"{time_old:.3f} s",
            f"{time_new:.3f} s",
            f"{time_ratio:,.1f}x faster",
        ),
        row("Peak RAM (MB)", f"{ram_old:.2f} MB", f"{ram_new:.2f} MB", f"{ram_ratio:,.1f}x less"),
        row("Articles processed", str(count_old), str(count_new), ""),
        row("Root hashes match", "", "", "YES" if hashes_match else "NO — MISMATCH"),
        f"└{sep}┘",
    ]
    return "\n".join(lines)


# ===========================================================================
# Subprocess entry point (called by each isolated trial)
# ===========================================================================


def _run_benchmark_mode(mode: str, file_path: str) -> None:
    """
    Single-mode entry point invoked inside an isolated subprocess per trial.

    Prints a JSON line to stdout with keys: label, time, ram, root, articles.
    The parent process parses these lines to aggregate trial results.
    """
    import json

    path = Path(file_path)
    if mode == "old":
        result = _run_old_way(path)
    elif mode == "new":
        result = _run_new_way(path)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    label, elapsed, ram, root, articles = result
    print(
        json.dumps(
            {
                "label": label,
                "time": elapsed,
                "ram": ram,
                "root": root,
                "articles": articles,
            }
        )
    )


# ===========================================================================
# Main entry point
# ===========================================================================


def run_benchmark(file_path: str, trials: int = 3) -> None:
    """
    Execute both benchmarks across multiple isolated trials with alternating
    order to minimise OS pagecache and allocator warm-up bias.

    Each trial spawns a fresh Python subprocess so memory and file-cache state
    are fully isolated between measurements.  The order of old-vs-new is
    reversed on odd-numbered trials.  Median time and peak-RAM across all
    trials are reported.

    Parameters
    ----------
    file_path:
        Path to the Wikipedia ``.xml.bz2`` dump.
    trials:
        Number of measurement trials (default 3; must be odd for a clean median).
    """
    import json
    import statistics
    import subprocess

    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    old_times: List[float] = []
    old_rams: List[float] = []
    new_times: List[float] = []
    new_rams: List[float] = []
    old_root: Optional[str] = None
    new_root: Optional[str] = None
    old_articles = 0
    new_articles = 0

    print(f"\nRunning {trials}-trial benchmark on: {path.name}")
    print("  Each trial runs in an isolated subprocess to avoid pagecache bias.\n")

    for i in range(trials):
        # Alternate order: even trials run old→new, odd trials run new→old
        order = ["old", "new"] if i % 2 == 0 else ["new", "old"]

        for mode in order:
            label = "OLD WAY" if mode == "old" else "NEW WAY"
            print(f"  Trial {i + 1}/{trials} — {label} …", end=" ", flush=True)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "openverifiablellm.benchmark",
                    "--_mode",
                    mode,
                    file_path,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"\n[ERROR] Subprocess failed (mode={mode}):\n{proc.stderr}", file=sys.stderr)
                sys.exit(1)

            # Extract the last non-empty line that parses as valid JSON.
            # This tolerates any stray log/warning lines on stdout that may
            # appear before or after the single JSON payload line.
            _stdout_lines = proc.stdout.splitlines()
            data = None
            for _line in reversed(_stdout_lines):
                _line = _line.strip()
                if _line:
                    try:
                        data = json.loads(_line)
                        break
                    except json.JSONDecodeError:
                        continue
            if data is None:
                print(
                    f"\n[ERROR] Could not find valid JSON in subprocess output "
                    f"(mode={mode}):\n{proc.stdout}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"time={data['time']:.3f}s  ram={data['ram']:.2f}MB")

            if mode == "old":
                old_times.append(data["time"])
                old_rams.append(data["ram"])
                old_root = data["root"]
                old_articles = data["articles"]
            else:
                new_times.append(data["time"])
                new_rams.append(data["ram"])
                new_root = data["root"]
                new_articles = data["articles"]

    # Abort if either run found zero articles — spurious matching roots.
    if old_articles == 0 or new_articles == 0:
        print(
            f"\n[ERROR] Zero articles processed "
            f"(old={old_articles}, new={new_articles}). "
            "Cannot produce meaningful benchmark results.",
            file=sys.stderr,
        )
        sys.exit(1)

    old_result: BenchmarkResult = (
        "Old Way (in-memory)",
        round(statistics.median(old_times), 3),
        round(statistics.median(old_rams), 2),
        old_root,
        old_articles,
    )
    new_result: BenchmarkResult = (
        "New Way (streaming)",
        round(statistics.median(new_times), 3),
        round(statistics.median(new_rams), 2),
        new_root,
        new_articles,
    )

    # Print terminal table
    print()
    print(_render_terminal_table(old_result, new_result, path.name, trials))

    # Print GitHub-Flavored Markdown table
    md = _render_markdown_table(old_result, new_result, path.name, trials)
    print("\n" + "=" * 60)
    print("Copy the block below into your GitHub Pull Request:")
    print("=" * 60)
    print(md)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Before-vs-After benchmark: in-memory vs streaming Merkle tree "
            "on a Wikipedia XML.bz2 dump."
        )
    )
    parser.add_argument(
        "file_path",
        help="Path to the Wikipedia XML.bz2 dump file (e.g. simplewiki-20260201-....xml.bz2)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of isolated subprocess trials (default: 3)",
    )
    # Internal flag used by the subprocess runner — not intended for direct use.
    parser.add_argument("--_mode", choices=["old", "new"], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args._mode:
        # Running as a subprocess trial — just measure and print JSON.
        _run_benchmark_mode(args._mode, args.file_path)
    else:
        run_benchmark(args.file_path, trials=args.trials)


if __name__ == "__main__":
    main()
