"""
test_fingerprint.py

Integration tests for the environment fingerprint feature.

Run with:
    pytest tests/test_fingerprint.py -v -s
"""

import json
from openverifiablellm.environment import generate_environment_fingerprint
from openverifiablellm.utils import generate_manifest


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