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
import defusedxml.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from openverifiablellm.incremental_merkle import IncrementalMerkleTree
from openverifiablellm.utils import clean_wikitext, extract_text_from_xml

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for a benchmark result row
# ---------------------------------------------------------------------------
BenchmarkResult = Tuple[
    str,   # approach label
    float, # wall-clock seconds
    float, # peak RAM in MB
    str,   # root hash (hex)
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

    with bz2.open(file_path, "rb") as raw:
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
    # Hash each article text to a leaf
    leaves: List[bytes] = [
        hashlib.sha256(t.encode("utf-8")).digest() for t in all_texts
    ]

    # Batch construction: classic bottom-up Merkle tree
    if not leaves:
        root_hex = hashlib.sha256(b"").hexdigest()
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

    stream = extract_text_from_xml(file_path, stream=True)
    assert stream is not None, "extract_text_from_xml must return a generator when stream=True"
    for article_text in stream:
        tree.append_leaf(article_text)

    root_hex: str = tree.get_root_hash() or hashlib.sha256(b"").hexdigest()

    t_end = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        "New Way (streaming)",
        round(t_end - t_start, 3),
        round(_bytes_to_mb(peak_bytes), 2),
        root_hex,
    )


# ===========================================================================
# Reporting: GitHub-Flavored Markdown table
# ===========================================================================

def _render_markdown_table(
    old: BenchmarkResult,
    new: BenchmarkResult,
    file_name: str,
) -> str:
    """
    Render a GFM Markdown table suitable for direct use in a GitHub PR.

    Calculates speed-up and RAM reduction ratios and appends a legend.
    """
    label_old, time_old, ram_old, hash_old = old
    label_new, time_new, ram_new, hash_new = new

    # Guard against division by zero on extremely fast runs
    time_ratio = (time_old / time_new) if time_new > 0 else float("inf")
    ram_ratio  = (ram_old  / ram_new)  if ram_new  > 0 else float("inf")

    hashes_match = (hash_old == hash_new)
    hash_verdict = "YES — identical root hash" if hashes_match else "NO  — MISMATCH (investigate!)"

    lines = [
        "",
        "## Benchmark Results",
        "",
        f"> **Input file:** `{file_name}`  ",
        f"> **Root hashes match:** {hash_verdict}",
        "",
        "| Metric                        | Old Way (in-memory) | New Way (streaming) | Improvement        |",
        "|-------------------------------|--------------------:|--------------------:|--------------------|",
        f"| Wall-clock time (s)           | `{time_old:>10.3f}` | `{time_new:>10.3f}` | **{time_ratio:,.1f}× faster**    |",
        f"| Peak RAM usage (MB)           | `{ram_old:>10.2f}` | `{ram_new:>10.2f}` | **{ram_ratio:,.1f}× less RAM**   |",
        f"| Root hash                     | `{hash_old[:16]}…` | `{hash_new[:16]}…` | {'Match' if hashes_match else 'MISMATCH'}              |",
        "",
        "### Notes",
        "- *Peak RAM* is measured with `tracemalloc` (Python heap only; does not include",
        "  OS-level buffers or the bzip2 decompressor's internal state).",
        "- *Wall-clock time* is measured with `time.perf_counter` on a single run.",
        "  For publication-quality numbers repeat 3× and report median ± std-dev.",
        "- The Old Way intentionally omits `elem.clear()` to reproduce the OOM behaviour.",
        "- The New Way uses `extract_text_from_xml(..., stream=True)` + `IncrementalMerkleTree` from this PR.",
        "",
    ]
    return "\n".join(lines)


def _render_terminal_table(
    old: BenchmarkResult,
    new: BenchmarkResult,
    file_name: str,
) -> str:
    """Plain-text box table for terminal output (complements the Markdown table)."""
    label_old, time_old, ram_old, hash_old = old
    label_new, time_new, ram_new, hash_new = new
    time_ratio = (time_old / time_new) if time_new > 0 else float("inf")
    ram_ratio  = (ram_old  / ram_new)  if ram_new  > 0 else float("inf")
    hashes_match = hash_old == hash_new

    w = 90
    sep = "─" * w

    def row(col1: str, col2: str, col3: str, col4: str = "") -> str:
        return f"│ {col1:<28} │ {col2:>18} │ {col3:>18} │ {col4:<14} │"

    lines = [
        f"┌{sep}┐",
        f"│{'BEFORE vs. AFTER  —  ' + file_name:^{w}}│",
        f"├{sep}┤",
        row("Metric", "Old Way", "New Way", "Improvement"),
        f"├{sep}┤",
        row("Wall-clock time (s)", f"{time_old:.3f} s", f"{time_new:.3f} s", f"{time_ratio:,.1f}x faster"),
        row("Peak RAM (MB)", f"{ram_old:.2f} MB", f"{ram_new:.2f} MB", f"{ram_ratio:,.1f}x less"),
        row("Root hashes match", "", "", "YES" if hashes_match else "NO — MISMATCH"),
        f"└{sep}┘",
    ]
    return "\n".join(lines)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_benchmark(file_path: str) -> None:
    """
    Execute both benchmarks sequentially and print the results.

    Parameters
    ----------
    file_path:
        Path to the Wikipedia ``.xml.bz2`` dump.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning OLD WAY benchmark on: {path.name}")
    print("  (This may take several minutes and use significant RAM …)\n")
    old_result = _run_old_way(path)
    print(f"  Done. Time={old_result[1]:.3f}s  Peak RAM={old_result[2]:.2f} MB")

    print(f"\nRunning NEW WAY benchmark on: {path.name}")
    print("  (Streaming — should use constant, minimal RAM …)\n")
    new_result = _run_new_way(path)
    print(f"  Done. Time={new_result[1]:.3f}s  Peak RAM={new_result[2]:.2f} MB")

    # Print terminal table
    print()
    print(_render_terminal_table(old_result, new_result, path.name))

    # Print GitHub-Flavored Markdown table
    md = _render_markdown_table(old_result, new_result, path.name)
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
    args = parser.parse_args(argv)
    run_benchmark(args.file_path)


if __name__ == "__main__":
    main()
