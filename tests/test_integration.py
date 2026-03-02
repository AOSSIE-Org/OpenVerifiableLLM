import json
import tempfile
from pathlib import Path
from openverifiablellm.utils import generate_manifest
from openverifiablellm.environment import generate_environment_fingerprint


def test_manifest_includes_environment():
    """Test that generate_manifest includes environment fingerprint"""
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as raw_file:
        raw_file.write("<test>raw data</test>")
        raw_path = raw_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as processed_file:
        processed_file.write('{"processed": "data"}')
        processed_path = processed_file.name
    
    try:
        # Generate manifest
        generate_manifest(raw_path, processed_path)
        
        # Check if manifest file was created
        manifest_path = Path.cwd() / "data" / "dataset_manifest.json"
        assert manifest_path.exists(), "Manifest file not created"
        
        # Read and verify manifest content
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Verify environment fields are present
        assert "environment" in manifest, "Environment field missing from manifest"
        assert "environment_hash" in manifest, "Environment hash missing from manifest"
        
        # Verify environment data structure
        env = manifest["environment"]
        assert "python_version" in env, "Python version missing"
        assert "platform" in env, "Platform missing"
        assert "pip_packages" in env, "Pip packages missing"
        
        # Verify hash is valid (64 hex chars)
        env_hash = manifest["environment_hash"]
        assert len(env_hash) == 64, "Environment hash should be 64 characters"
        assert all(c in '0123456789abcdef' for c in env_hash), "Hash should be hex"
        
        print("✅ Integration test passed!")
        print(f"✅ Environment hash: {env_hash[:16]}...")
        print(f"✅ Manifest contains {len(env)} environment fields")
        
    finally:
        # Cleanup
        Path(raw_path).unlink(missing_ok=True)
        Path(processed_path).unlink(missing_ok=True)
        Path(manifest_path).unlink(missing_ok=True)


def test_environment_fingerprint_consistency():
    """Test that environment fingerprint is consistent across calls"""
    
    env1 = generate_environment_fingerprint()
    env2 = generate_environment_fingerprint()
    
    # Hashes should be identical for same environment
    assert env1["environment_hash"] == env2["environment_hash"], "Environment hashes should be consistent"
    
    # Environment data should be identical
    assert env1["environment"] == env2["environment"], "Environment data should be consistent"
    
    print("✅ Environment fingerprint consistency test passed!")
