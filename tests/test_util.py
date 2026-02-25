import bz2
import hashlib
import pytest
from openverifiablellm import utils

"""
Unit and integration tests for OpenVerifiableLLM preprocessing pipeline.

Run with:
    pip install -e ".[dev]"
    pytest
"""

# --------------- clean_wikitext tests ------------------------------------

def test_clean_wikitext_removes_templates_and_refs():
    text = "Hello {{Infobox}} <ref>cite</ref> world"
    cleaned = utils.clean_wikitext(text)
    assert cleaned == "Hello world"


def test_clean_wikitext_handles_links():
    text = "This is [[Python|programming language]] and [[India]]"
    cleaned = utils.clean_wikitext(text)
    assert cleaned == "This is programming language and India"


def test_clean_wikitext_collapses_whitespace():
    text = "Hello   world\n\n   test"
    cleaned = utils.clean_wikitext(text)
    assert cleaned == "Hello world test"
    
# --------------- extract_dump_date tests ------------------------------------

def test_extract_dump_date_valid():
    filename = "simplewiki-20260201-pages-articles.xml.bz2"
    assert utils.extract_dump_date(filename) == "2026-02-01"


def test_extract_dump_date_invalid():
    filename = "no-date-file.xml.bz2"
    assert utils.extract_dump_date(filename) == "unknown"

# --------------- generate manifest ------------------------------------

def test_generate_manifest_raises_if_processed_missing(tmp_path):
    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("dummy")

    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        utils.generate_manifest(raw_file, missing_file)
        
def test_generate_manifest_runs_if_file_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("dummy")

    processed_file = tmp_path / "processed.txt"
    processed_file.write_text("cleaned")

    utils.generate_manifest(raw_file, processed_file)

    manifest_file = tmp_path / "data/dataset_manifest.json"
    assert manifest_file.exists()
    
# --------------- compute_sha256 ------------------------------------

def test_correct_sha256_output(tmp_path):
    # Create a temporary file
    file = tmp_path / "sample.txt"
    content = "hello wikipedia"
    file.write_text(content, encoding="utf-8")

    # Expected hash using standard hashlib
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Hash using your function
    actual = utils.compute_sha256(str(file))

    # Verify correctness
    assert actual == expected


def test_different_content_different_hash(tmp_path):
    file1 = tmp_path / "content_a.txt"
    file2 = tmp_path / "content_b.txt"

    file1.write_text("Content A", encoding="utf-8")
    file2.write_text("Content B", encoding="utf-8")

    assert utils.compute_sha256(file1) != utils.compute_sha256(file2)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        utils.compute_sha256("non_existent_file.txt")
        
# --------------- extract_text_from_xml tests ------------------------------------

def test_extract_text_from_xml_end_to_end(tmp_path, monkeypatch):

    xml_content = """<?xml version="1.0"?>
    <mediawiki>
      <page>
        <revision>
          <text>Hello [[World]]</text>
        </revision>
      </page>
    </mediawiki>
    """

    input_file = tmp_path / "simplewiki-20260201-pages.xml.bz2"

    with bz2.open(input_file, "wt", encoding="utf-8") as f:
        f.write(xml_content)

    # Redirect project root
    monkeypatch.chdir(tmp_path)

    utils.extract_text_from_xml(input_file)

    processed_file = tmp_path / "data/processed/wiki_clean.txt"
    assert processed_file.exists()

    assert "Hello World" in processed_file.read_text()
    
    # --------------- manifest includes merkle fields ------------------------------------

def test_manifest_contains_merkle_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("dummy data")

    processed_file = tmp_path / "processed.txt"
    processed_file.write_text("cleaned data")

    utils.generate_manifest(raw_file, processed_file)

    manifest_file = tmp_path / "data/dataset_manifest.json"
    manifest = manifest_file.read_text()

    assert "raw_merkle_root" in manifest
    assert "processed_merkle_root" in manifest
    assert "chunk_size_bytes" in manifest