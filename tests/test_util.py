import bz2
import hashlib
import json

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


def test_compute_sha256_bytes(tmp_path):
    file = tmp_path / "sample.txt"
    content = b"hello wikipedia"
    file.write_bytes(content)

    expected = hashlib.sha256(content).digest()

    actual_data = utils.compute_sha256_bytes(data=content)
    actual_file = utils.compute_sha256_bytes(file_path=file)

    assert actual_data == expected
    assert actual_file == expected


def test_correct_sha256_output(tmp_path):
    file = tmp_path / "sample.txt"
    content = "hello wikipedia"
    file.write_text(content, encoding="utf-8")

    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Hash using your function
    actual = utils.compute_sha256(file_path=str(file))

    assert actual == expected


def test_different_content_different_hash(tmp_path):
    file1 = tmp_path / "content_a.txt"
    file2 = tmp_path / "content_b.txt"

    file1.write_text("Content A", encoding="utf-8")
    file2.write_text("Content B", encoding="utf-8")

    assert utils.compute_sha256(file_path=file1) != utils.compute_sha256(file_path=file2)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        utils.compute_sha256(file_path="non_existent_file.txt")


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

    monkeypatch.chdir(tmp_path)

    utils.extract_text_from_xml(input_file)

    processed_file = tmp_path / "data/processed/wiki_clean.txt"
    assert processed_file.exists()

    assert "Hello World" in processed_file.read_text()


def test_extract_text_from_xml_uncompressed(tmp_path, monkeypatch):
    xml_content = """<?xml version="1.0"?>
    <mediawiki>
      <page>
        <revision>
          <text>Hello [[Uncompressed]]</text>
        </revision>
      </page>
    </mediawiki>
    """

    input_file = tmp_path / "simplewiki-20260201-pages.xml"

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Redirect project root
    monkeypatch.chdir(tmp_path)

    utils.extract_text_from_xml(input_file)

    processed_file = tmp_path / "data/processed/wiki_clean.txt"
    assert processed_file.exists()

    assert "Hello Uncompressed" in processed_file.read_text()


# --------------- manifest includes merkle fields ------------------------------------


def test_manifest_contains_merkle_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("dummy data")

    processed_file = tmp_path / "processed.txt"
    processed_file.write_text("cleaned data")

    utils.generate_manifest(raw_file, processed_file)

    manifest_file = tmp_path / "data/dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())

    assert "raw_merkle_root" in manifest
    assert "processed_merkle_root" in manifest
    assert "chunk_size_bytes" in manifest


# --------------- compute_merkle_root ------------------------------------


def test_merkle_root_deterministic(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("hello wikipedia")

    root1 = utils.compute_merkle_root(file, chunk_size=4)
    root2 = utils.compute_merkle_root(file, chunk_size=4)

    assert root1 == root2


def test_merkle_root_changes_when_content_changes(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("content A")

    root1 = utils.compute_merkle_root(file)

    file.write_text("content B")

    root2 = utils.compute_merkle_root(file)

    assert root1 != root2


def test_merkle_root_single_chunk_equals_sha256(tmp_path):
    file = tmp_path / "data.txt"
    content = "small file"
    file.write_text(content)

    merkle_root = utils.compute_merkle_root(file, chunk_size=10_000)

    expected = hashlib.sha256(content.encode()).hexdigest()

    assert merkle_root == expected


def test_merkle_root_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("")

    root = utils.compute_merkle_root(file)

    expected = hashlib.sha256(b"").hexdigest()

    assert root == expected


def test_compute_merkle_root_multi_chunk_hardcoded(tmp_path):
    file = tmp_path / "data.txt"
    # 3 chunks of 8 bytes each
    chunk1 = b"chunk__1"
    chunk2 = b"chunk__2"
    chunk3 = b"chunk__3"
    file.write_bytes(chunk1 + chunk2 + chunk3)

    h1 = hashlib.sha256(chunk1).digest()
    h2 = hashlib.sha256(chunk2).digest()
    h3 = hashlib.sha256(chunk3).digest()

    h12 = hashlib.sha256(h1 + h2).digest()
    h33 = hashlib.sha256(h3 + h3).digest()

    expected_root = hashlib.sha256(h12 + h33).hexdigest()

    actual_root = utils.compute_merkle_root(file, chunk_size=8)
    assert actual_root == expected_root


# --------------- Merkle proof generation ------------------------------------


def test_merkle_proof_verification(tmp_path):
    file = tmp_path / "data.txt"
    content = b"hello world this is merkle proof test"
    file.write_bytes(content)

    root = utils.compute_merkle_root(file, chunk_size=8)
    proof = utils.generate_merkle_proof(file, chunk_index=1, chunk_size=8)

    with file.open("rb") as f:
        f.seek(8)
        chunk = f.read(8)

    assert utils.verify_merkle_proof(chunk, proof, root)

    tampered_chunk = bytearray(chunk)
    tampered_chunk[0] ^= 1
    assert not utils.verify_merkle_proof(bytes(tampered_chunk), proof, root)

    bad_proof = proof.copy()
    bad_proof[0] = ("00" * 32, proof[0][1])
    assert not utils.verify_merkle_proof(chunk, bad_proof, root)


def test_export_and_load_merkle_proof(tmp_path):
    file = tmp_path / "data.txt"
    content = b"portable proof verification example"
    file.write_bytes(content)

    root = utils.compute_merkle_root(file, chunk_size=8)
    proof = utils.generate_merkle_proof(file, chunk_index=1, chunk_size=8)

    proof_file = tmp_path / "proof.json"

    utils.export_merkle_proof(proof, chunk_index=1, chunk_size=8, output_path=proof_file)

    with file.open("rb") as f:
        f.seek(8)
        chunk = f.read(8)

    assert utils.verify_merkle_proof_from_file(
        proof_file_path=proof_file,
        chunk_data=chunk,
        expected_root=root,
    )


def test_extract_text_from_xml_malformed_xml(tmp_path, monkeypatch):
    import defusedxml.ElementTree as ET

    malformed_xml_content = """<?xml version="1.0"?>
    <mediawiki>
      <page>
        <revision>
          <text>Hello [[Malformed]]
        </revision>
      </page>
    </mediawiki>
    """

    input_file = tmp_path / "simplewiki-20260201-pages-malformed.xml"

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(malformed_xml_content)

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ET.ParseError):
        utils.extract_text_from_xml(input_file)
