"""Negative tests for the saved OVL segment-slice prototype.

Runs against temporary copies of saved packages. Original package files are
never modified. Synthetic-data research only.
"""
from __future__ import annotations

import copy
import importlib
import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

import prototype_saved_slice as prototype


def main() -> None:
    source = HERE / "saved_packages" / "segment_1.json"
    original = json.loads(source.read_text(encoding="utf-8"))

    mutations = {
        "corrupted token data": lambda p: p["windows"][0]["tokens"].__setitem__(
            0, p["windows"][0]["tokens"][0] + 1
        ),
        "reordered windows": lambda p: p["windows"].reverse(),
        "incomplete windows": lambda p: p["windows"].pop(),
    }

    original_package_dir = prototype.PACKAGE_DIR
    try:
        for label, mutate in mutations.items():
            package = copy.deepcopy(original)
            mutate(package)

            with tempfile.TemporaryDirectory(prefix="ovl-slice-negative-") as tmp:
                package_dir = Path(tmp)
                (package_dir / "segment_1.json").write_text(
                    json.dumps(package, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )
                prototype.PACKAGE_DIR = package_dir

                try:
                    with redirect_stdout(StringIO()):
                        prototype.verify()
                except Exception as exc:
                    print(f"PASS: rejected {label} ({type(exc).__name__}: {exc})")
                else:
                    raise AssertionError(
                        f"Verifier ACCEPTED {label}; expected rejection"
                    )
    finally:
        prototype.PACKAGE_DIR = original_package_dir

    print("SUCCESS: all three tampering cases were rejected.")
    print("Original saved_packages files were not modified.")


if __name__ == "__main__":
    main()
