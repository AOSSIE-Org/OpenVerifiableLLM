import pytest
from scripts.generate_manifest import generate_manifest

def test_generate_manifest_raises_if_processed_missing(tmp_path):
    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("dummy")

    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        generate_manifest(raw_file, missing_file)
        
def test_generate_manifest_runs_if_file_exists(tmp_path):
    raw_file = tmp_path / "raw.txt"
    raw_file.write_text("dummy")

    processed_file = tmp_path / "processed.txt"
    processed_file.write_text("cleaned")

    # Should not raise
    generate_manifest(raw_file, processed_file)