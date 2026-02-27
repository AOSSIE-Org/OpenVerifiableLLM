import json
import hashlib
from pathlib import Path
from typing import Union, Dict
from .utils import compute_sha256


def normalize_line_endings(content: str) -> str:
    return content.replace('\r\n', '\n').replace('\r', '\n')


def compute_normalized_sha256(file_path: Union[str, Path]) -> str:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    normalized_content = normalize_line_endings(content)
    sha256 = hashlib.sha256()
    sha256.update(normalized_content.encode("utf-8"))
    return sha256.hexdigest()


def generate_manifest(directory_path: Union[str, Path]) -> Dict:
    dir_path = Path(directory_path)
    files = [f for f in dir_path.glob("**/*") if f.is_file()]
    files.sort(key=lambda x: str(x.relative_to(dir_path)).replace("\\", "/"))
    
    manifest_entries = []
    for file in files:
        rel_path = str(file.relative_to(dir_path)).replace("\\", "/")
        try:
            file_hash = compute_normalized_sha256(file)
        except (UnicodeDecodeError, ValueError):
            file_hash = compute_sha256(file)
        
        manifest_entries.append({
            "path": rel_path,
            "sha256": file_hash,
            "size": file.stat().st_size
        })
    
    return {"files": manifest_entries}


def get_manifest_hash(manifest: Dict) -> str:
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def run_pipeline(directory_path: Union[str, Path]) -> str:
    manifest = generate_manifest(directory_path)
    return get_manifest_hash(manifest)


def validate_manifest_integrity(manifest: Dict, directory_path: Union[str, Path]) -> bool:
    dir_path = Path(directory_path)
    for entry in manifest["files"]:
        file_path = dir_path / entry["path"]
        try:
            recomputed_hash = compute_normalized_sha256(file_path)
        except (UnicodeDecodeError, ValueError):
            recomputed_hash = compute_sha256(file_path)
        if entry["sha256"] != recomputed_hash:
            return False
    return True


def compare_manifests(manifest1: Dict, manifest2: Dict) -> Dict[str, list]:
    files1 = {entry["path"]: entry["sha256"] for entry in manifest1.get("files", [])}
    files2 = {entry["path"]: entry["sha256"] for entry in manifest2.get("files", [])}
    
    all_paths = set(files1.keys()) | set(files2.keys())
    
    result = {"added": [], "removed": [], "modified": [], "unchanged": []}
    
    for path in sorted(all_paths):
        if path in files1 and path not in files2:
            result["removed"].append(path)
        elif path not in files1 and path in files2:
            result["added"].append(path)
        elif files1[path] != files2[path]:
            result["modified"].append(path)
        else:
            result["unchanged"].append(path)
    
    return result
