"""
Tests for Manifest Chain (tamper detection) feature.

Tests the cryptographic linking of dataset manifests where each manifest
includes the SHA256 hash of its predecessor, enabling tamper detection.

Run with:
    python -m pytest tests/test_manifest_chain.py -v
"""

import json

import pytest

from openverifiablellm.manifest_chain import (
    _canonical_json,
    compute_manifest_hash,
    get_parent_manifest_hash,
    verify_manifest_chain,
    verify_manifest_chain_link,
)


class TestCanonicalJson:
    """Tests for canonical JSON serialization."""

    def test_canonical_json_consistent_ordering(self):
        """Same object serialized twice should be identical."""
        obj = {"z": 1, "a": 2, "m": 3}
        
        json1 = _canonical_json(obj)
        json2 = _canonical_json(obj)
        
        assert json1 == json2

    def test_canonical_json_key_order_normalized(self):
        """Different key orders should produce the same output."""
        obj1 = {"z": 1, "a": 2}
        obj2 = {"a": 2, "z": 1}
        
        assert _canonical_json(obj1) == _canonical_json(obj2)

    def test_canonical_json_no_spaces(self):
        """Canonical JSON should have no extra spaces."""
        obj = {"key": "value"}
        result = _canonical_json(obj)
        
        assert " " not in result
        assert result == '{"key":"value"}'

    def test_canonical_json_nested_objects(self):
        """Nested objects should also be canonically ordered."""
        obj = {"outer": {"z": 1, "a": 2}}
        result = _canonical_json(obj)
        
        # The inner object keys should be sorted
        assert '"a":2' in result
        assert '"z":1' in result
        # And a should come before z
        assert result.index('"a":') < result.index('"z":')


class TestComputeManifestHash:
    """Tests for compute_manifest_hash function."""

    def test_hash_from_dict(self):
        """Should compute hash from manifest dict."""
        manifest = {"key": "value", "number": 42}
        hash_val = compute_manifest_hash(manifest)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex string length

    def test_hash_from_file(self, tmp_path):
        """Should compute hash from manifest JSON file."""
        manifest = {"key": "value"}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))
        
        hash_val = compute_manifest_hash(manifest_file)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_hash_deterministic_dict(self):
        """Same dict should always hash to same value."""
        manifest = {"a": 1, "b": 2}
        
        hash1 = compute_manifest_hash(manifest)
        hash2 = compute_manifest_hash(manifest)
        
        assert hash1 == hash2

    def test_hash_deterministic_different_key_order(self):
        """Different key order should produce same hash."""
        manifest1 = {"z": 1, "a": 2}
        manifest2 = {"a": 2, "z": 1}
        
        assert compute_manifest_hash(manifest1) == compute_manifest_hash(manifest2)

    def test_hash_excludes_parent_manifest_hash(self):
        """Computing hash should ignore parent_manifest_hash field."""
        manifest1 = {"key": "value"}
        manifest2 = {"key": "value", "parent_manifest_hash": "abc123"}
        
        # Both should hash the same since parent_manifest_hash is excluded
        assert compute_manifest_hash(manifest1) == compute_manifest_hash(manifest2)

    def test_hash_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            compute_manifest_hash(tmp_path / "nonexistent.json")

    def test_hash_malformed_json(self, tmp_path):
        """Should raise ValueError if JSON is malformed."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")
        
        with pytest.raises(ValueError):
            compute_manifest_hash(bad_file)

    def test_hash_changes_on_content_change(self):
        """Hash should change when manifest content changes."""
        manifest1 = {"key": "value1"}
        manifest2 = {"key": "value2"}
        
        assert compute_manifest_hash(manifest1) != compute_manifest_hash(manifest2)


class TestGetParentManifestHash:
    """Tests for get_parent_manifest_hash function."""

    def test_returns_empty_string_if_file_missing(self, tmp_path):
        """Should return empty string if manifest doesn't exist yet."""
        missing_path = tmp_path / "nonexistent.json"
        
        result = get_parent_manifest_hash(missing_path)
        
        assert result == ""

    def test_returns_hash_if_file_exists(self, tmp_path):
        """Should return hash of existing manifest."""
        manifest = {"key": "value"}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))
        
        result = get_parent_manifest_hash(manifest_file)
        
        # Should be a valid SHA256 hash
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_value_matches_computed(self, tmp_path):
        """Returned hash should match independently computed hash."""
        manifest = {"key": "value"}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))
        
        returned_hash = get_parent_manifest_hash(manifest_file)
        computed_hash = compute_manifest_hash(manifest_file)
        
        assert returned_hash == computed_hash


class TestVerifyManifestChainLink:
    """Tests for verify_manifest_chain_link function."""

    def test_valid_chain_link(self, tmp_path):
        """Should verify a correct chain link."""
        # Create previous manifest
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        # Create current manifest with correct parent hash
        prev_hash = compute_manifest_hash(prev_manifest)
        current_manifest = {"version": 2, "parent_manifest_hash": prev_hash}
        
        assert verify_manifest_chain_link(prev_file, current_manifest)

    def test_broken_chain_link_wrong_hash(self, tmp_path):
        """Should fail if parent hash doesn't match."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        # Current manifest with wrong parent hash
        current_manifest = {"version": 2, "parent_manifest_hash": "0" * 64}
        
        assert not verify_manifest_chain_link(prev_file, current_manifest)

    def test_broken_chain_link_missing_hash(self, tmp_path):
        """Should fail if current manifest lacks parent_manifest_hash."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        # Current manifest without parent_manifest_hash
        current_manifest = {"version": 2}
        
        # Should treat missing field as empty string, which won't match
        assert not verify_manifest_chain_link(prev_file, current_manifest)

    def test_chain_link_with_file_path(self, tmp_path):
        """Should accept file paths for both previous and current manifests."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        prev_hash = compute_manifest_hash(prev_file)
        current_manifest = {"version": 2, "parent_manifest_hash": prev_hash}
        current_file = tmp_path / "manifest_v2.json"
        current_file.write_text(json.dumps(current_manifest))
        
        assert verify_manifest_chain_link(prev_file, current_file)

    def test_chain_link_previous_file_missing(self, tmp_path):
        """Should raise FileNotFoundError if previous manifest doesn't exist."""
        current_manifest = {"version": 2, "parent_manifest_hash": "abc"}
        
        with pytest.raises(FileNotFoundError):
            verify_manifest_chain_link(tmp_path / "missing.json", current_manifest)

    def test_chain_link_current_file_missing(self, tmp_path):
        """Should raise FileNotFoundError if current manifest file doesn't exist."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        with pytest.raises(FileNotFoundError):
            verify_manifest_chain_link(prev_file, tmp_path / "missing.json")

    def test_chain_link_current_malformed(self, tmp_path):
        """Should raise ValueError if current manifest is malformed."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid")
        
        with pytest.raises(ValueError):
            verify_manifest_chain_link(prev_file, bad_file)


class TestVerifyManifestChain:
    """Tests for verify_manifest_chain function."""

    def test_chain_report_structure(self, tmp_path):
        """Report should have expected keys."""
        manifest = {"key": "value"}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))
        
        report = verify_manifest_chain(manifest_file)
        
        assert "chain_valid" in report
        assert "chain_message" in report
        assert "has_parent_hash_field" in report
        assert "parent_hash_value" in report

    def test_missing_manifest(self, tmp_path):
        """Should report invalid for missing manifest."""
        report = verify_manifest_chain(tmp_path / "nonexistent.json")
        
        assert report["chain_valid"] is False
        assert "not found" in report["chain_message"]

    def test_first_run_manifest_empty_parent(self, tmp_path):
        """First-run manifest with empty parent_manifest_hash should be valid."""
        manifest = {
            "key": "value",
            "parent_manifest_hash": ""
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))
        
        report = verify_manifest_chain(manifest_file)
        
        assert report["chain_valid"] is True
        assert report["has_parent_hash_field"] is True
        assert report["parent_hash_value"] == ""

    def test_old_manifest_no_parent_field(self, tmp_path):
        """Old manifest without parent_manifest_hash field."""
        manifest = {"key": "value"}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))
        
        report = verify_manifest_chain(manifest_file)
        
        assert report["has_parent_hash_field"] is False
        assert report["parent_hash_value"] == ""

    def test_with_previous_manifest_valid_link(self, tmp_path):
        """Should verify link when previous manifest is provided."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        prev_hash = compute_manifest_hash(prev_file)
        current_manifest = {"version": 2, "parent_manifest_hash": prev_hash}
        current_file = tmp_path / "manifest_v2.json"
        current_file.write_text(json.dumps(current_manifest))
        
        report = verify_manifest_chain(current_file, previous_manifest_path=prev_file)
        
        assert report["chain_valid"] is True

    def test_with_previous_manifest_broken_link(self, tmp_path):
        """Should report broken link when provided."""
        prev_manifest = {"version": 1}
        prev_file = tmp_path / "manifest_v1.json"
        prev_file.write_text(json.dumps(prev_manifest))
        
        current_manifest = {"version": 2, "parent_manifest_hash": "0" * 64}
        current_file = tmp_path / "manifest_v2.json"
        current_file.write_text(json.dumps(current_manifest))
        
        report = verify_manifest_chain(current_file, previous_manifest_path=prev_file)
        
        assert report["chain_valid"] is False
        assert "broken" in report["chain_message"].lower()


class TestIntegrationChainSequence:
    """Integration tests for a sequence of manifests."""

    def test_three_manifest_chain(self, tmp_path):
        """Verify a chain of three manifests."""
        # Manifest 1 (first run)
        m1 = {"run": 1}
        m1_file = tmp_path / "manifest_1.json"
        m1_file.write_text(json.dumps(m1))
        
        m1_hash = compute_manifest_hash(m1_file)
        
        # Manifest 2 (references 1)
        m2 = {"run": 2, "parent_manifest_hash": m1_hash}
        m2_file = tmp_path / "manifest_2.json"
        m2_file.write_text(json.dumps(m2))
        
        m2_hash = compute_manifest_hash(m2_file)
        
        # Manifest 3 (references 2)
        m3 = {"run": 3, "parent_manifest_hash": m2_hash}
        m3_file = tmp_path / "manifest_3.json"
        m3_file.write_text(json.dumps(m3))
        
        # Verify each link
        assert verify_manifest_chain_link(m1_file, m2)
        assert verify_manifest_chain_link(m2_file, m3)
        
        # Verify m3 knows about m2
        report3 = verify_manifest_chain(m3_file)
        assert report3["has_parent_hash_field"] is True
        assert report3["parent_hash_value"] == m2_hash

    def test_tampered_manifest_breaks_chain(self, tmp_path):
        """Modifying an earlier manifest breaks the chain."""
        # Create chain: M1 -> M2 -> M3
        m1 = {"run": 1}
        m1_file = tmp_path / "manifest_1.json"
        m1_file.write_text(json.dumps(m1))
        
        m1_hash = compute_manifest_hash(m1_file)
        m2 = {"run": 2, "parent_manifest_hash": m1_hash}
        m2_file = tmp_path / "manifest_2.json"
        m2_file.write_text(json.dumps(m2))
        
        m2_hash = compute_manifest_hash(m2_file)
        m3 = {"run": 3, "parent_manifest_hash": m2_hash}
        m3_file = tmp_path / "manifest_3.json"
        m3_file.write_text(json.dumps(m3))
        
        # Verify chain is intact
        assert verify_manifest_chain_link(m2_file, m3)
        
        # Now tamper with M2
        tampered_m2 = {"run": 2, "parent_manifest_hash": m1_hash, "tampered": True}
        m2_file.write_text(json.dumps(tampered_m2))
        
        # Chain should now be broken
        assert not verify_manifest_chain_link(m2_file, m3)


class TestBackwardCompatibility:
    """Tests for backward compatibility with old manifests."""

    def test_old_manifest_without_chain_field(self, tmp_path):
        """Old manifests without parent_manifest_hash should still work."""
        # Simulate an old manifest
        old_manifest = {
            "wikipedia_dump": "test.xml.bz2",
            "raw_sha256": "abc123",
            "processed_sha256": "def456",
            # Note: no parent_manifest_hash field
        }
        old_file = tmp_path / "old_manifest.json"
        old_file.write_text(json.dumps(old_manifest))
        
        report = verify_manifest_chain(old_file)
        
        # Should report as non-chain-aware but not fail
        assert report["has_parent_hash_field"] is False
        assert report["chain_valid"] is False

    def test_adding_chain_to_old_manifest(self, tmp_path):
        """Should be able to add chain support by adding parent_manifest_hash."""
        # Old manifest
        old_manifest = {"run": 1}
        old_file = tmp_path / "manifest_old.json"
        old_file.write_text(json.dumps(old_manifest))
        
        old_hash = compute_manifest_hash(old_file)
        
        # New manifest adds parent reference
        new_manifest = {"run": 2, "parent_manifest_hash": old_hash}
        new_file = tmp_path / "manifest_new.json"
        new_file.write_text(json.dumps(new_manifest))
        
        # Chain should be verifiable
        assert verify_manifest_chain_link(old_file, new_manifest)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
