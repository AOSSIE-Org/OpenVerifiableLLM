
"""Negative tests for the saved OVL segment-slice prototype."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prototype_saved_slice as prototype  # noqa: E402


def verify_from(package_dir: Path) -> None:
    original_dir = prototype.PACKAGE_DIR
    try:
        prototype.PACKAGE_DIR = package_dir
        with redirect_stdout(StringIO()):
            prototype.verify()
    finally:
        prototype.PACKAGE_DIR = original_dir


def write_package(package_dir: Path, package: dict) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "segment_1.json").write_text(
        json.dumps(package, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    source = HERE / "saved_packages" / "segment_1.json"
    original = json.loads(source.read_text(encoding="utf-8"))

    # Control case: an unchanged package must verify successfully.
    with tempfile.TemporaryDirectory(prefix="ovl-slice-baseline-") as tmp:
        baseline_dir = Path(tmp)
        write_package(baseline_dir, copy.deepcopy(original))
        verify_from(baseline_dir)

    print("PASS: unchanged package was accepted.")

    mutations = {
        "corrupted token data": (
            "digest mismatch",
            lambda p: p["windows"][0]["tokens"].__setitem__(
                0, p["windows"][0]["tokens"][0] + 1
            ),
        ),
        "reordered windows": (
            "position mismatch",
            lambda p: p["windows"].reverse(),
        ),
        "incomplete windows": (
            "position mismatch",
            lambda p: p["windows"].pop(),
        ),
    }

    for label, (expected_error, mutate) in mutations.items():
        package = copy.deepcopy(original)
        mutate(package)

        with tempfile.TemporaryDirectory(prefix="ovl-slice-negative-") as tmp:
            package_dir = Path(tmp)
            write_package(package_dir, package)

            try:
                verify_from(package_dir)
            except AssertionError as exc:
                if expected_error not in str(exc):
                    raise AssertionError(
                        f"Unexpected verifier failure for {label}: {exc}"
                    ) from exc

                print(f"PASS: rejected {label} ({exc})")
            else:
                raise AssertionError(
                    f"Verifier ACCEPTED {label}; expected rejection"
                )

    print("SUCCESS: all three tampering cases were rejected.")


if __name__ == "__main__":
    main()
