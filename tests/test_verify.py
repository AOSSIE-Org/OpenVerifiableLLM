"""
Tests for Deterministic Preprocessing Verification Mode.

Uses only the standard library (unittest) — no pytest needed.

Run with:
    python -m pytest tests/test_verify.py -v
    # or
    python tests/test_verify.py -v
"""

import bz2
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from openverifiablellm import utils
from openverifiablellm.verify import (
    CheckResult,
    CheckStatus,
    VerificationReport,
    _check_field,
    _load_manifest,
    verify_preprocessing,
)

# Shared fixture helpers
XML_CONTENT = """<?xml version="1.0"?>
<mediawiki>
  <page>
    <revision>
      <text>Hello [[World]] this is a test page.</text>
    </revision>
  </page>
  <page>
    <revision>
      <text>Another page about [[Python|programming]].</text>
    </revision>
  </page>
</mediawiki>
"""


def make_dump(tmp_dir: Path) -> Path:
    """Write a minimal compressed Wikipedia dump and return its path."""
    dump = tmp_dir / "simplewiki-20260201-pages-articles.xml.bz2"
    with bz2.open(dump, "wt", encoding="utf-8") as f:
        f.write(XML_CONTENT)
    return dump


def run_preprocessing(tmp_dir: Path, dump: Path) -> None:
    """Run utils.extract_text_from_xml() with tmp_dir as the working directory."""
    original = os.getcwd()
    os.chdir(tmp_dir)
    try:
        utils.extract_text_from_xml(dump)
    finally:
        os.chdir(original)


class TmpMixin(unittest.TestCase):
    """Creates and cleans up a fresh temporary directory for each test."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# VerificationReport unit tests


class TestVerificationReport(unittest.TestCase):
    def _r(self):
        return VerificationReport(input_dump="d.bz2", manifest_path="m.json")

    def test_all_passed_vacuously_true_with_no_checks(self):
        self.assertTrue(self._r().all_passed)

    def test_all_passed_true_when_only_passes(self):
        r = self._r()
        r.add(CheckResult("a", CheckStatus.PASS))
        r.add(CheckResult("b", CheckStatus.PASS))
        self.assertTrue(r.all_passed)

    def test_all_passed_false_when_any_failure(self):
        r = self._r()
        r.add(CheckResult("a", CheckStatus.PASS))
        r.add(CheckResult("b", CheckStatus.FAIL, expected="x", actual="y"))
        self.assertFalse(r.all_passed)

    def test_counts_segregated_correctly(self):
        r = self._r()
        r.add(CheckResult("p1", CheckStatus.PASS))
        r.add(CheckResult("p2", CheckStatus.PASS))
        r.add(CheckResult("f1", CheckStatus.FAIL))
        r.add(CheckResult("s1", CheckStatus.SKIP))
        self.assertEqual(len(r.passed), 2)
        self.assertEqual(len(r.failed), 1)
        self.assertEqual(len(r.skipped), 1)

    def test_summary_verdict_pass(self):
        r = self._r()
        r.add(CheckResult("x", CheckStatus.PASS))
        self.assertIn("ALL CHECKS PASSED", r.summary())

    def test_summary_verdict_fail(self):
        r = self._r()
        r.add(CheckResult("x", CheckStatus.FAIL, expected="a", actual="b"))
        self.assertIn("VERIFICATION FAILED", r.summary())

    def test_summary_mentions_input_dump(self):
        self.assertIn("d.bz2", self._r().summary())

    def test_to_dict_json_serialisable(self):
        r = self._r()
        r.add(CheckResult("x", CheckStatus.PASS))
        json.dumps(r.to_dict())  # must not raise

    def test_to_dict_count_matches_checks(self):
        r = self._r()
        r.add(CheckResult("a", CheckStatus.PASS))
        r.add(CheckResult("b", CheckStatus.FAIL))
        d = r.to_dict()
        self.assertEqual(d["counts"]["total"], 2)
        self.assertEqual(d["counts"]["passed"], 1)
        self.assertEqual(d["counts"]["failed"], 1)

    def test_to_dict_each_check_has_required_keys(self):
        r = self._r()
        r.add(CheckResult("x", CheckStatus.PASS, expected="a", actual="a", detail="note"))
        for check in r.to_dict()["checks"]:
            for key in ("name", "status", "expected", "actual", "detail"):
                self.assertIn(key, check)

    def test_failed_check_str_shows_expected_and_actual(self):
        c = CheckResult("h", CheckStatus.FAIL, expected="abc", actual="xyz")
        s = str(c)
        self.assertIn("abc", s)
        self.assertIn("xyz", s)

    def test_check_status_icons(self):
        self.assertIn("✓", str(CheckResult("x", CheckStatus.PASS)))
        self.assertIn("✗", str(CheckResult("x", CheckStatus.FAIL)))
        self.assertIn("~", str(CheckResult("x", CheckStatus.SKIP)))


# _check_field unit tests
class TestCheckField(unittest.TestCase):
    def _r(self):
        return VerificationReport(input_dump="d", manifest_path="m")

    def test_equal_strings_produce_pass(self):
        r = self._r()
        _check_field(r, "sha", "abc", "abc")
        self.assertEqual(r.checks[-1].status, CheckStatus.PASS)

    def test_unequal_strings_produce_fail(self):
        r = self._r()
        _check_field(r, "sha", "abc", "xyz")
        c = r.checks[-1]
        self.assertEqual(c.status, CheckStatus.FAIL)
        self.assertEqual(c.expected, "abc")
        self.assertEqual(c.actual, "xyz")

    def test_non_string_types_coerced_and_compared(self):
        r = self._r()
        _check_field(r, "v", 42, 42)
        self.assertEqual(r.checks[-1].status, CheckStatus.PASS)

    def test_detail_stored_on_result(self):
        r = self._r()
        _check_field(r, "k", "a", "a", detail="my note")
        self.assertEqual(r.checks[-1].detail, "my note")


# _load_manifest unit tests


class TestLoadManifest(TmpMixin):
    def test_raises_file_not_found_if_absent(self):
        with self.assertRaises(FileNotFoundError):
            _load_manifest(self.tmp / "no.json")

    def test_error_message_contains_filename(self):
        try:
            _load_manifest(self.tmp / "missing_manifest.json")
        except FileNotFoundError as e:
            self.assertIn("missing_manifest.json", str(e))

    def test_loads_valid_json(self):
        p = self.tmp / "m.json"
        p.write_text(json.dumps({"k": "v"}))
        self.assertEqual(_load_manifest(p)["k"], "v")


# Integration: happy path


class TestHappyPath(TmpMixin):
    def setUp(self):
        super().setUp()
        self.dump = make_dump(self.tmp)
        run_preprocessing(self.tmp, self.dump)

    def test_all_checks_pass(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        self.assertTrue(r.all_passed, r.summary())

    def test_required_check_names_present(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        names = {c.name for c in r.checks}
        for required in (
            "manifest_exists",
            "raw_file_exists",
            "raw_sha256",
            "processed_sha256",
            "dump_date",
            "wikipedia_dump_name",
            "python_version",
            "reprocessing_succeeded",
        ):
            self.assertIn(required, names)

    def test_raw_sha256_check_passes(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "raw_sha256")
        self.assertEqual(c.status, CheckStatus.PASS)

    def test_processed_sha256_check_passes(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "processed_sha256")
        self.assertEqual(c.status, CheckStatus.PASS)

    def test_merkle_checks_pass_or_skip(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        merkle = [c for c in r.checks if "merkle" in c.name]
        self.assertGreaterEqual(len(merkle), 1)
        for c in merkle:
            self.assertIn(c.status, (CheckStatus.PASS, CheckStatus.SKIP))

    def test_dump_date_check_passes(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "dump_date")
        self.assertEqual(c.status, CheckStatus.PASS)

    def test_explicit_manifest_path_accepted(self):
        mp = self.tmp / "data" / "dataset_manifest.json"
        r = verify_preprocessing(self.dump, manifest_path=mp)
        self.assertTrue(r.all_passed, r.summary())

    def test_report_stores_input_dump_path(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        self.assertEqual(r.input_dump, str(self.dump))


# Integration: failure scenarios


class TestFailureScenarios(TmpMixin):
    def setUp(self):
        super().setUp()
        self.dump = make_dump(self.tmp)
        run_preprocessing(self.tmp, self.dump)

    def _manifest_path(self):
        return self.tmp / "data" / "dataset_manifest.json"

    def _read_manifest(self):
        return json.loads(self._manifest_path().read_text())

    def _write_manifest(self, data):
        self._manifest_path().write_text(json.dumps(data, indent=2))

    def test_fails_when_manifest_absent(self):
        empty_root = Path(tempfile.mkdtemp())
        try:
            r = verify_preprocessing(self.dump, project_root=empty_root)
            c = next(x for x in r.checks if x.name == "manifest_exists")
            self.assertEqual(c.status, CheckStatus.FAIL)
        finally:
            shutil.rmtree(empty_root)

    def test_fails_when_input_dump_missing(self):
        r = verify_preprocessing(self.tmp / "ghost.xml.bz2", project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "raw_file_exists")
        self.assertEqual(c.status, CheckStatus.FAIL)

    def test_fails_when_raw_file_tampered(self):
        """A tampered .bz2 must cause failure on raw_sha256 or reprocessing_succeeded."""
        td = Path(tempfile.mkdtemp())
        try:
            tampered = td / self.dump.name
            tampered.write_bytes(self.dump.read_bytes()[:-10] + b"CORRUPTED!")
            r = verify_preprocessing(tampered, project_root=self.tmp)
            self.assertFalse(r.all_passed)
            relevant = [
                c
                for c in r.checks
                if c.name in ("raw_sha256", "reprocessing_succeeded")
                and c.status == CheckStatus.FAIL
            ]
            self.assertGreater(len(relevant), 0, "Expected at least one failure for tampered file")
        finally:
            shutil.rmtree(td)

    def test_fails_when_processed_sha256_wrong_in_manifest(self):
        m = self._read_manifest()
        m["processed_sha256"] = "0" * 64
        self._write_manifest(m)
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "processed_sha256")
        self.assertEqual(c.status, CheckStatus.FAIL)

    def test_fails_when_raw_sha256_wrong_in_manifest(self):
        m = self._read_manifest()
        m["raw_sha256"] = "1" * 64
        self._write_manifest(m)
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "raw_sha256")
        self.assertEqual(c.status, CheckStatus.FAIL)

    def test_fails_when_processed_merkle_wrong(self):
        m = self._read_manifest()
        m["processed_merkle_root"] = "f" * 64
        self._write_manifest(m)
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "processed_merkle_root")
        self.assertEqual(c.status, CheckStatus.FAIL)

    def test_fails_when_raw_merkle_wrong(self):
        m = self._read_manifest()
        m["raw_merkle_root"] = "a" * 64
        self._write_manifest(m)
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        c = next(x for x in r.checks if x.name == "raw_merkle_root")
        self.assertEqual(c.status, CheckStatus.FAIL)

    def test_fails_when_dump_name_differs(self):
        td = Path(tempfile.mkdtemp())
        try:
            wrong = td / "simplewiki-99991231-pages-articles.xml.bz2"
            shutil.copy(self.dump, wrong)
            r = verify_preprocessing(wrong, project_root=self.tmp)
            c = next(x for x in r.checks if x.name == "wikipedia_dump_name")
            self.assertEqual(c.status, CheckStatus.FAIL)
        finally:
            shutil.rmtree(td)


# Legacy manifest compatibility (no Merkle fields)
class TestLegacyManifest(TmpMixin):
    def setUp(self):
        super().setUp()
        self.dump = make_dump(self.tmp)
        run_preprocessing(self.tmp, self.dump)
        mp = self.tmp / "data" / "dataset_manifest.json"
        m = json.loads(mp.read_text())
        for k in ("raw_merkle_root", "processed_merkle_root", "chunk_size_bytes"):
            m.pop(k, None)
        mp.write_text(json.dumps(m, indent=2))

    def test_merkle_checks_are_skipped(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        for name in ("raw_merkle_root", "processed_merkle_root"):
            c = next((x for x in r.checks if x.name == name), None)
            self.assertIsNotNone(c, f"check '{name}' not found")
            self.assertEqual(c.status, CheckStatus.SKIP)

    def test_other_checks_still_pass(self):
        r = verify_preprocessing(self.dump, project_root=self.tmp)
        failed = [c for c in r.checks if c.status == CheckStatus.FAIL]
        self.assertEqual(failed, [], f"Unexpected failures: {failed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
