"""
benchmark.py  (scripts/)
========================
CLI entry-point for the before-vs-after benchmark.

This thin wrapper delegates all logic to
``openverifiablellm.benchmark.main``, keeping the scripts/ directory as
a collection of plain launchers with no duplicated implementation.

Usage
-----
    # Download the dump first (≈350 MB):
    python scripts/download_dump.py --wiki simplewiki --date 20260201

    # Then run the benchmark:
    python scripts/benchmark.py simplewiki-20260201-pages-articles-multistream.xml.bz2

    # Or, equivalently, via the package module:
    python -m openverifiablellm.benchmark <path>

What it measures
----------------
* **Old Way** (in-memory): decompress everything, collect all article texts
  in a list, build a batch Merkle tree — O(N) RAM.
* **New Way** (streaming): yield one article at a time with
  ``stream_text_from_xml``, feed each into an ``IncrementalMerkleTree``
  — O(log N) RAM.

The script prints:
1. A terminal box-table with wall-clock time and peak RAM side-by-side.
2. A GitHub-Flavored Markdown table you can paste straight into the PR.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the script is run directly
# (i.e. ``python scripts/benchmark.py``), even without an editable install.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openverifiablellm.benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
