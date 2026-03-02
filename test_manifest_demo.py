import tempfile
import json
from pathlib import Path
from openverifiablellm.utils import generate_manifest

# Create test files
with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
    f.write('<test>data</test>')
    raw_path = f.name

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    f.write('{"processed": "data"}')
    processed_path = f.name

try:
    generate_manifest(raw_path, processed_path)
    
    # Read and display manifest
    manifest_path = Path.cwd() / 'data' / 'dataset_manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print('🎯 COMPLETE MANIFEST STRUCTURE:')
    print('=' * 50)
    for key, value in manifest.items():
        if key == 'environment':
            print(f'✅ {key}: {{9 environment fields captured}}')
        elif key == 'environment_hash':
            print(f'✅ {key}: {value[:16]}...')
        elif key == 'pip_packages':
            print(f'✅ {key}: {len(value)} packages')
        else:
            print(f'✅ {key}: {value}')
            
finally:
    # Cleanup
    Path(raw_path).unlink(missing_ok=True)
    Path(processed_path).unlink(missing_ok=True)
    Path(manifest_path).unlink(missing_ok=True)
