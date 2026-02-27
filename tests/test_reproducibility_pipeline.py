import json
import hashlib
import tempfile
import pytest
from pathlib import Path
from openverifiablellm.dataset_hash import compute_sha256


def normalize_line_endings(content: str) -> str:
    return content.replace('\r\n', '\n').replace('\r', '\n')


def generate_manifest(directory_path: Path) -> dict:
    files = [f for f in directory_path.glob("**/*") if f.is_file()]
    files.sort(key=lambda x: str(x.relative_to(directory_path)).replace("\\", "/"))
    
    manifest_entries = []
    for file in files:
        rel_path = str(file.relative_to(directory_path)).replace("\\", "/")
        file_hash = compute_sha256(file)
        manifest_entries.append({
            "path": rel_path,
            "sha256": file_hash,
            "size": file.stat().st_size
        })
    
    return {"files": manifest_entries}


def get_manifest_hash(manifest: dict) -> str:
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def test_manifest_generation(tmp_path):
    (tmp_path / "file1.txt").write_text("Content 1", encoding="utf-8")
    (tmp_path / "file2.txt").write_text("Content 2", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("Nested content", encoding="utf-8")
    
    manifest = generate_manifest(tmp_path)
    
    assert "files" in manifest
    assert len(manifest["files"]) == 3
    
    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == ["file1.txt", "file2.txt", "subdir/nested.txt"]


def test_manifest_deterministic_ordering(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    
    (dir1 / "z.txt").write_text("z content", encoding="utf-8")
    (dir1 / "a.txt").write_text("a content", encoding="utf-8")
    (dir1 / "m.txt").write_text("m content", encoding="utf-8")
    
    (dir2 / "a.txt").write_text("a content", encoding="utf-8")
    (dir2 / "m.txt").write_text("m content", encoding="utf-8")
    (dir2 / "z.txt").write_text("z content", encoding="utf-8")
    
    manifest1 = generate_manifest(dir1)
    manifest2 = generate_manifest(dir2)
    
    manifest1_json = json.dumps(manifest1, sort_keys=True, separators=(",", ":"))
    manifest2_json = json.dumps(manifest2, sort_keys=True, separators=(",", ":"))
    
    assert manifest1_json == manifest2_json


def test_manifest_hash_stability(tmp_path):
    (tmp_path / "data.txt").write_text("Test data", encoding="utf-8")
    
    manifest = generate_manifest(tmp_path)
    hash1 = get_manifest_hash(manifest)
    hash2 = get_manifest_hash(manifest)
    
    assert hash1 == hash2
    assert len(hash1) == 64


def test_manifest_changes_detected(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Version 1", encoding="utf-8")
    
    manifest1 = generate_manifest(tmp_path)
    hash1 = get_manifest_hash(manifest1)
    
    file_path.write_text("Version 2", encoding="utf-8")
    
    manifest2 = generate_manifest(tmp_path)
    hash2 = get_manifest_hash(manifest2)
    
    assert hash1 != hash2


def test_empty_directory_manifest(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    manifest = generate_manifest(empty_dir)
    assert manifest == {"files": []}
    
    hash_val = get_manifest_hash(manifest)
    expected = hashlib.sha256(b'{"files":[]}').hexdigest()
    assert hash_val == expected


def test_large_dataset_manifest(tmp_path):
    for i in range(100):
        file_path = tmp_path / f"file_{i:03d}.txt"
        file_path.write_text(f"Content {i}", encoding="utf-8")
    
    manifest = generate_manifest(tmp_path)
    assert len(manifest["files"]) == 100
    
    paths = [entry["path"] for entry in manifest["files"]]
    sorted_paths = sorted(paths)
    assert paths == sorted_paths


def test_cross_platform_path_normalization(tmp_path):
    (tmp_path / "test.txt").write_text("test", encoding="utf-8")
    
    manifest = generate_manifest(tmp_path)
    entry = manifest["files"][0]
    
    assert "/" in entry["path"] or entry["path"] == "test.txt"
    assert "\\" not in entry["path"]


def test_manifest_integrity(tmp_path):
    (tmp_path / "integrity.txt").write_text("Integrity test", encoding="utf-8")
    
    manifest = generate_manifest(tmp_path)
    
    for entry in manifest["files"]:
        file_path = tmp_path / entry["path"]
        recomputed_hash = compute_sha256(file_path)
        assert entry["sha256"] == recomputed_hash


def test_json_serialization_consistency():
    test_data = {
        "files": [
            {"path": "a.txt", "sha256": "a" * 64, "size": 100},
            {"path": "b.txt", "sha256": "b" * 64, "size": 200}
        ]
    }
    
    json1 = json.dumps(test_data, sort_keys=True, separators=(",", ":"))
    json2 = json.dumps(test_data, sort_keys=True, separators=(",", ":"))
    
    assert json1 == json2
    assert "\n" not in json1
    assert " " not in json1


def test_pipeline_integration(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "part1.txt").write_text("Dataset part 1", encoding="utf-8")
    (dataset_dir / "part2.txt").write_text("Dataset part 2", encoding="utf-8")
    (dataset_dir / "subdir").mkdir()
    (dataset_dir / "subdir" / "nested.txt").write_text("Nested data", encoding="utf-8")
    
    manifest = generate_manifest(dataset_dir)
    final_hash = get_manifest_hash(manifest)
    
    assert len(final_hash) == 64
    assert isinstance(final_hash, str)
    
    manifest2 = generate_manifest(dataset_dir)
    final_hash2 = get_manifest_hash(manifest2)
    assert final_hash == final_hash2


def test_file_addition_detection(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "file1.txt").write_text("Original", encoding="utf-8")
    manifest1 = generate_manifest(dataset_dir)
    hash1 = get_manifest_hash(manifest1)
    
    (dataset_dir / "file2.txt").write_text("New file", encoding="utf-8")
    manifest2 = generate_manifest(dataset_dir)
    hash2 = get_manifest_hash(manifest2)
    
    assert hash1 != hash2


def test_file_deletion_detection(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "file1.txt").write_text("File 1", encoding="utf-8")
    (dataset_dir / "file2.txt").write_text("File 2", encoding="utf-8")
    
    manifest1 = generate_manifest(dataset_dir)
    hash1 = get_manifest_hash(manifest1)
    
    (dataset_dir / "file2.txt").unlink()
    manifest2 = generate_manifest(dataset_dir)
    hash2 = get_manifest_hash(manifest2)
    
    assert hash1 != hash2
    assert len(manifest2["files"]) == 1


def test_binary_files_in_manifest(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "text.txt").write_text("Text content", encoding="utf-8")
    
    binary_data = bytes(range(256))
    (dataset_dir / "binary.bin").write_bytes(binary_data)
    
    manifest = generate_manifest(dataset_dir)
    assert len(manifest["files"]) == 2
    
    binary_path = dataset_dir / "binary.bin"
    binary_hash = compute_sha256(binary_path)
    
    for entry in manifest["files"]:
        if entry["path"] == "binary.bin":
            assert entry["sha256"] == binary_hash
            break
    else:
        pytest.fail("Binary file not found in manifest")
