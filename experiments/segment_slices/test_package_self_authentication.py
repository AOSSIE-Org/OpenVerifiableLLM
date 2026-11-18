"""Demonstrate the current prototype's self-authentication limitation.

This intentionally edits a temporary copy of segment_1.json, changes one token,
and recomputes the expected digest stored in that same package. It should be
accepted by the current verifier, demonstrating why an external trusted
commitment/signature is needed. Original saved packages are never modified.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prototype_saved_slice as prototype


def main() -> None:
    source = HERE / "saved_packages" / "segment_1.json"
    package = json.loads(source.read_text(encoding="utf-8"))

    old_value = package["windows"][0]["tokens"][0]
    package["windows"][0]["tokens"][0] = old_value + 1

    # Recompute the expected digest from the altered content and store it
    # inside the same package, exactly the weakness this test investigates.
    batches = prototype.reconstruct(package["windows"])
    package["expected_batches"] = [
        {"sha256": prototype.batch_digest(batch), "after": after}
        for batch, after in batches
    ]

    original_dir = prototype.PACKAGE_DIR
    try:
        with tempfile.TemporaryDirectory(prefix="ovl-slice-self-auth-") as tmp:
            temp_dir = Path(tmp)
            (temp_dir / "segment_1.json").write_text(
                json.dumps(package, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            prototype.PACKAGE_DIR = temp_dir
            prototype.verify()
    finally:
        prototype.PACKAGE_DIR = original_dir

    print(
        f"CONFIRMED: verifier accepted changed token {old_value} -> "
        f"{package['windows'][0]['tokens'][0]} after its in-package digest "
        "was recomputed."
    )
    print("This is expected for the current prototype, not a test failure.")
    print("Original saved_packages files were not modified.")
    print(
        "Conclusion: the prototype checks internal consistency, but does not "
        "authenticate package contents against a trusted external commitment."
    )


if __name__ == "__main__":
    main()
