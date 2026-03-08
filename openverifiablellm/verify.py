"""
Deterministic Preprocessing Verification Mode
==============================================

Re-runs dataset preprocessing from scratch and validates that all
generated artifacts (SHA256, Merkle roots, manifest fields) match
the previously recorded manifest exactly.

Usage (CLI):
    python -m openverifiablellm.verify <input_dump> [--manifest <path>]

Usage (Python):
    from openverifiablellm.verify import verify_preprocessing
    report = verify_preprocessing("data/simplewiki-20260201.xml.bz2")
    print(report.summary())
"""

import json
import logging
import platform
import tempfile
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import sys
import subprocess
import os

from openverifiablellm import utils

logger = logging.getLogger(__name__)

class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    expected: Optional[str] = None
    actual: Optional[str] = None
    detail: Optional[str] = None

    def __str__(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "~"}[self.status.value]
        base = f"  [{icon}] {self.name}: {self.status.value}"
        if self.status == CheckStatus.FAIL:
            base += f"\n        expected : {self.expected}"
            base += f"\n        actual   : {self.actual}"
        if self.detail:
            base += f"\n        note     : {self.detail}"
        return base


@dataclass
class VerificationReport:
    """
    Holds the full set of check results from a verification run.
    """
    input_dump: str
    manifest_path: str
    checks: list[CheckResult] = field(default_factory=list)


    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        log = logger.info if check.status == CheckStatus.PASS else logger.warning
        log("Check %s → %s", check.name, check.status.value)

    @property
    def passed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.PASS]

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def skipped(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.SKIP]

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0


    def summary(self) -> str:
        # Table width
        width = 110

        def line(char="─"):
            return char * width

        def center(text):
            return text.center(width)

        def row(col1, col2="", col3="", col4=""):
            return f"│ {col1:<28} │ {col2:<10} │ {col3:<28} │ {col4:<28} │"

        lines = []

        # Header
        lines.append("┌" + line("─") + "┐")
        lines.append("│" + center("DETERMINISTIC PREPROCESSING VERIFICATION REPORT") + "│")
        lines.append("├" + line("─") + "┤")

        # Metadata section
        lines.append(f"│ Input Dump : {self.input_dump:<88}│")
        lines.append(f"│ Manifest   : {self.manifest_path:<88}│")
        lines.append("├" + line("─") + "┤")

        # Summary counts
        counts = f"Total: {len(self.checks)}   ✓ {len(self.passed)}   ✗ {len(self.failed)}   ~ {len(self.skipped)}"
        lines.append("│" + center(counts) + "│")
        lines.append("├" + line("─") + "┤")

        # Table header
        lines.append(row("Check Name", "Status", "Expected", "Actual"))
        lines.append("├" + line("─") + "┤")

        # Rows
        for check in self.checks:
            expected = (check.expected or "")[:28]
            actual = (check.actual or "")[:28]
            lines.append(row(check.name[:28], check.status.value, expected, actual))

        lines.append("├" + line("─") + "┤")

        # Final result
        verdict = "ALL CHECKS PASSED" if self.all_passed else "VERIFICATION FAILED"
        lines.append("│" + center(verdict) + "│")
        lines.append("└" + line("─") + "┘")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "input_dump": self.input_dump,
            "manifest_path": self.manifest_path,
            "all_passed": self.all_passed,
            "counts": {
                "total": len(self.checks),
                "passed": len(self.passed),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "expected": c.expected,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


# Core helpers

def _check_field(
    report: VerificationReport,
    name: str,
    expected,
    actual,
    detail: Optional[str] = None,
) -> None:
    """Compare a single manifest field and record the result."""
    exp_str = str(expected)
    act_str = str(actual)
    status = CheckStatus.PASS if exp_str == act_str else CheckStatus.FAIL
    report.add(CheckResult(name=name, status=status,
                           expected=exp_str, actual=act_str, detail=detail))


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run preprocessing first to generate it."
        )
    with manifest_path.open() as f:
        return json.load(f)


# Public API
def verify_preprocessing(
    input_dump: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    *,
    project_root: Optional[Path] = None,
) -> VerificationReport:
    """
    Re-run preprocessing on *input_dump* in an isolated temp directory and
    validate every artifact against the stored manifest.

    Parameters
    ----------
    input_dump :
        Path to the raw Wikipedia XML (or .bz2) dump.
    manifest_path :
        Explicit path to ``dataset_manifest.json``.
        Defaults to ``<project_root>/data/dataset_manifest.json``.
    project_root :
        Root used to locate the default manifest.
        Defaults to ``Path.cwd()``.

    Returns
    -------
    VerificationReport
        Structured report with per-check pass/fail results.
    """
    input_dump = Path(input_dump).resolve()
    root = project_root or Path.cwd()

    if manifest_path is None:
        manifest_path = root / "data" / "dataset_manifest.json"
    else:
        manifest_path = Path(manifest_path)

    report = VerificationReport(
        input_dump=str(input_dump),
        manifest_path=str(manifest_path),
    )

    # 1. Load existing manifest
    try:
        manifest = _load_manifest(manifest_path)
    except FileNotFoundError as exc:
        report.add(CheckResult(
            name="manifest_exists",
            status=CheckStatus.FAIL,
            detail=str(exc),
        ))
        return report

    report.add(CheckResult(
        name="manifest_exists",
        status=CheckStatus.PASS,
        detail=str(manifest_path),
    ))

    # 2. Validate raw file integrity BEFORE re-processing
    if not input_dump.exists():
        report.add(CheckResult(
            name="raw_file_exists",
            status=CheckStatus.FAIL,
            detail=f"Input dump not found: {input_dump}",
        ))
        return report

    report.add(CheckResult(name="raw_file_exists", status=CheckStatus.PASS))

    # SHA256 of raw file
    raw_sha256_actual = utils.compute_sha256(file_path=input_dump)
    _check_field(
        report, "raw_sha256",
        expected=manifest.get("raw_sha256"),
        actual=raw_sha256_actual,
        detail="SHA256 of the raw input dump",
    )

    # Merkle root of raw file
    if "raw_merkle_root" in manifest:
        chunk_size = manifest.get("chunk_size_bytes", utils.MERKLE_CHUNK_SIZE_BYTES)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            report.add(CheckResult(
                name="chunk_size_bytes",
                status=CheckStatus.FAIL,
                expected=str(utils.MERKLE_CHUNK_SIZE_BYTES),
                actual=str(chunk_size),
                detail="Manifest chunk_size_bytes must be a positive integer",
            ))
            return report
        raw_merkle_actual = utils.compute_merkle_root(input_dump, chunk_size=chunk_size)
        _check_field(
            report, "raw_merkle_root",
            expected=manifest["raw_merkle_root"],
            actual=raw_merkle_actual,
            detail=f"Merkle root of raw dump (chunk={chunk_size} bytes)",
        )
    else:
        report.add(CheckResult(
            name="raw_merkle_root",
            status=CheckStatus.SKIP,
            detail="Field absent from manifest (older version)",
        ))

    # 3. Metadata / environment checks
    _check_field(
        report, "dump_date",
        expected=manifest.get("dump_date"),
        actual=utils.extract_dump_date(input_dump.name),
        detail="Dump date parsed from filename",
    )

    _check_field(
        report, "wikipedia_dump_name",
        expected=manifest.get("wikipedia_dump"),
        actual=input_dump.name,
        detail="Raw filename recorded in manifest",
    )

    python_ver = platform.python_version()
    expected_python = manifest.get("python_version")
    if expected_python and python_ver != expected_python:
        report.add(CheckResult(
            name="python_version",
            status=CheckStatus.FAIL,
            expected=expected_python,
            actual=python_ver,
            detail="Python version mismatch may cause non-deterministic output",
        ))
    else:
        report.add(CheckResult(
            name="python_version",
            status=CheckStatus.PASS,
            expected=expected_python,
            actual=python_ver,
        ))

    # 4. Re-run preprocessing in an isolated temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="ovllm_verify_"))
    try:
        logger.info("Re-running preprocessing in temp dir: %s", tmp_dir)

        try:

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

            subprocess.run(
                [sys.executable, "-m", "openverifiablellm.utils", str(input_dump)],
                cwd=tmp_dir,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            # Decompression or XML parse failure — tampered / corrupt file
            report.add(CheckResult(
                name="reprocessing_succeeded",
                status=CheckStatus.FAIL,
                detail=f"Re-run failed with exit code {exc.returncode}: {exc.stderr.strip()}",
            ))
            return report

        reproduced_processed = tmp_dir / "data" / "processed" / "wiki_clean.txt"

        if not reproduced_processed.exists():
            report.add(CheckResult(
                name="reprocessing_succeeded",
                status=CheckStatus.FAIL,
                detail="wiki_clean.txt was not produced during re-run",
            ))
            return report

        report.add(CheckResult(
            name="reprocessing_succeeded",
            status=CheckStatus.PASS,
            detail=str(reproduced_processed),
        ))

        # 5. Compare reproduced processed file against manifest

        # SHA256 of reproduced processed file
        proc_sha256_actual = utils.compute_sha256(file_path=reproduced_processed)
        _check_field(
            report, "processed_sha256",
            expected=manifest.get("processed_sha256"),
            actual=proc_sha256_actual,
            detail="SHA256 of reproduced wiki_clean.txt",
        )

        # Merkle root of reproduced processed file
        if "processed_merkle_root" in manifest:
            chunk_size = manifest.get("chunk_size_bytes", utils.MERKLE_CHUNK_SIZE_BYTES)
            proc_merkle_actual = utils.compute_merkle_root(
                reproduced_processed, chunk_size=chunk_size
            )
            _check_field(
                report, "processed_merkle_root",
                expected=manifest["processed_merkle_root"],
                actual=proc_merkle_actual,
                detail=f"Merkle root of reproduced processed file (chunk={chunk_size} bytes)",
            )
        else:
            report.add(CheckResult(
                name="processed_merkle_root",
                status=CheckStatus.SKIP,
                detail="Field absent from manifest (older version)",
            ))

        # 6. Compare reproduced manifest fields
        reproduced_manifest_path = tmp_dir / "data" / "dataset_manifest.json"
        if reproduced_manifest_path.exists():
            with reproduced_manifest_path.open() as f:
                reproduced_manifest = json.load(f)

            _check_field(
                report, "manifest_preprocessing_version",
                expected=manifest.get("preprocessing_version"),
                actual=reproduced_manifest.get("preprocessing_version"),
                detail="Preprocessing version tag",
            )
            if "chunk_size_bytes" in manifest:
                _check_field(
                    report, "manifest_chunk_size_bytes",
                    expected=manifest["chunk_size_bytes"],
                    actual=reproduced_manifest.get("chunk_size_bytes"),
                    detail="Merkle chunk size used during preprocessing",
                )
            else:
                report.add(CheckResult(
                    name="manifest_chunk_size_bytes",
                    status=CheckStatus.SKIP,
                    detail="Field absent from manifest (older version)",
                ))
        else:
            report.add(CheckResult(
                name="manifest_regenerated",
                status=CheckStatus.SKIP,
                detail="Reproduced manifest not found — skipping manifest field checks",
            ))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return report


# CLI entry point

def main(argv=None):
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Deterministic Preprocessing Verification Mode"
    )
    parser.add_argument("input_dump", help="Path to the raw Wikipedia XML (.bz2) dump")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to dataset_manifest.json (default: data/dataset_manifest.json)",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        default=None,
        help="Optional path to write the full report as JSON",
    )
    args = parser.parse_args(argv)

    report = verify_preprocessing(
        input_dump=args.input_dump,
        manifest_path=args.manifest,
    )

    print(report.summary())

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nJSON report written to: {out}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
