"""
tests/test_environment.py

Tests for the environment fingerprinting feature.
Covers fingerprint output, consistency, and manifest integration.

Run with:
    pytest tests/test_environment.py -v -s
"""

import json
import tempfile
from pathlib import Path
from openverifiablellm.environment import compute_object_hash, generate_environment_fingerprint
from openverifiablellm.utils import generate_manifest


# --------------- compute_object_hash tests ------------------------------------

def test_hash_deterministic():
    """Same object must always produce the same hash."""
    obj = {"a": 1, "b": 2}
    assert compute_object_hash(obj) == compute_object_hash(obj)


def test_hash_order_independent():
    """Hash must be identical regardless of key order."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"b": 2, "a": 1}
    assert compute_object_hash(obj1) == compute_object_hash(obj2)


def test_hash_changes_on_modification():
    """Changing a value must produce a different hash."""
    obj = {"a": 1}
    h1 = compute_object_hash(obj)
    obj["a"] = 2
    h2 = compute_object_hash(obj)
    assert h1 != h2


# --------------- generate_environment_fingerprint tests ------------------------------------

def test_fingerprint_returns_hash_and_environment():
    """Fingerprint must return both environment_hash and environment keys."""
    result = generate_environment_fingerprint()
    assert "environment_hash" in result
    assert "environment" in result


def test_fingerprint_hash_is_valid_sha256():
    """Hash must be a valid 64-character hex SHA-256 string."""
    result = generate_environment_fingerprint()
    hash_val = result["environment_hash"]
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert all(c in "0123456789abcdef" for c in hash_val)


def test_fingerprint_contains_required_fields():
    """Environment block must contain all expected fields."""
    result = generate_environment_fingerprint()
    env = result["environment"]
    for field in ["python_version", "platform", "pytorch_version", "cuda_version", "gpu_name", "pip_packages"]:
        assert field in env, f"Missing field: {field}"


def test_fingerprint_package_count(capsys):
    """Print package count only — not the full list."""
    result = generate_environment_fingerprint()
    packages = result["environment"]["pip_packages"]
    assert isinstance(packages, list)
    assert len(packages) > 0
    print(f"\n✅ {len(packages)} packages installed")


def test_fingerprint_is_consistent_across_calls():
    """Same environment must produce the same hash every time."""
    result1 = generate_environment_fingerprint()
    result2 = generate_environment_fingerprint()
    assert result1["environment_hash"] == result2["environment_hash"]
    assert result1["environment"] == result2["environment"]
    print("✅ Environment fingerprint consistency test passed!")


def test_fingerprint_is_embedded_in_manifest(tmp_path, monkeypatch):
    """Hash inside manifest must match direct fingerprint call."""
    monkeypatch.chdir(tmp_path)

    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("raw data")

    processed_file = tmp_path / "processed.txt"
    processed_file.write_text("processed data")

    fingerprint = generate_environment_fingerprint()
    generate_manifest(raw_file, processed_file)

    manifest = json.loads((tmp_path / "data" / "dataset_manifest.json").read_text())

    assert manifest["environment_hash"] == fingerprint["environment_hash"]
    print(f"\n✅ Manifest hash matches: {fingerprint['environment_hash'][:16]}...")


def test_fingerprint_summary_print(capsys):
    """Print a clean human-readable summary of the fingerprint."""
    result = generate_environment_fingerprint()
    env = result["environment"]

    print("\n========== ENVIRONMENT FINGERPRINT SUMMARY ==========")
    print(f"Hash     : {result['environment_hash']}")
    print(f"Python   : {env['python_version'].split()[0]}")
    print(f"Platform : {env['platform']}")
    print(f"PyTorch  : {env['pytorch_version']}")
    print(f"CUDA     : {env['cuda_version']}")
    print(f"GPU      : {env['gpu_name']}")
    print(f"Packages : {len(env['pip_packages'])} packages installed")
    print("=====================================================")

    captured = capsys.readouterr()
    assert "ENVIRONMENT FINGERPRINT SUMMARY" in captured.out
    assert "packages installed" in captured.out


# --------------- manifest integration tests ------------------------------------

def test_manifest_includes_environment():
    """Test that generate_manifest includes environment fingerprint."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as raw_file:
        raw_file.write("<test>raw data</test>")
        raw_path = raw_file.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as processed_file:
        processed_file.write('{"processed": "data"}')
        processed_path = processed_file.name

    manifest_path = Path.cwd() / "data" / "dataset_manifest.json"

    try:
        generate_manifest(raw_path, processed_path)

        assert manifest_path.exists(), "Manifest file not created"

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        assert "environment" in manifest, "Environment field missing from manifest"
        assert "environment_hash" in manifest, "Environment hash missing from manifest"

        env = manifest["environment"]
        assert "python_version" in env, "Python version missing"
        assert "platform" in env, "Platform missing"
        assert "pip_packages" in env, "Pip packages missing"

        env_hash = manifest["environment_hash"]
        assert len(env_hash) == 64, "Environment hash should be 64 characters"
        assert all(c in '0123456789abcdef' for c in env_hash), "Hash should be hex"

        print("✅ Integration test passed!")
        print(f"✅ Environment hash: {env_hash[:16]}...")
        print(f"✅ Manifest contains {len(env)} environment fields")

    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(processed_path).unlink(missing_ok=True)
        Path(manifest_path).unlink(missing_ok=True)