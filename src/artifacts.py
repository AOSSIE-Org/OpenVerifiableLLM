import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import torch

MERKLE_CHUNK_SIZE_BYTES = 1024 * 1024

# All generated artifacts (checkpoints, signatures, telemetry logs, manifests)
# land under RUNS_DIR, anchored at the repo root so the location is the same
# no matter which directory a script is launched from. Gitignored; override
# with OVL_RUNS_DIR for scratch/CI layouts.
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("OVL_RUNS_DIR", REPO_ROOT / "runs"))

CHECKPOINT_STATE_PATH = RUNS_DIR / "mid_checkpoint.pt"
CHECKPOINT_WEIGHTS_PATH = RUNS_DIR / "mid_checkpoint.safetensors"
CHECKPOINT_MERKLE_PATH = RUNS_DIR / "mid_checkpoint.merkle.json"


def hash_json(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_sha256_bytes(
    *,
    data: Optional[Union[bytes, bytearray]] = None,
    file_path: Optional[Union[str, Path]] = None,
) -> bytes:
    if (data is None) == (file_path is None):
        raise ValueError("Exactly one of data or file_path must be provided")

    h = hashlib.sha256()
    if data is not None:
        h.update(data)
        return h.digest()

    with Path(file_path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.digest()


def compute_sha256(
    *,
    data: Optional[Union[bytes, bytearray]] = None,
    file_path: Optional[Union[str, Path]] = None,
) -> str:
    return compute_sha256_bytes(data=data, file_path=file_path).hex()


def model_parameters_sha256(model: "torch.nn.Module") -> str:
    h = hashlib.sha256()
    for param in model.parameters():
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def tensor_mapping_sha256(tensors: Dict[str, "torch.Tensor"]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# Merkle tree constructions, named so a manifest is self-describing.
#
# "sha256-concat" is the original construction: leaves are the raw chunk
# digests and a parent is sha256(left + right). It is vulnerable to the
# classic second-preimage attack -- an internal node can be replayed as a
# leaf, so a shallower tree can be forged to the same root.
#
# "rfc6962-sha256" prefixes leaves with 0x00 and internal nodes with 0x01
# (Certificate Transparency, RFC 6962 section 2.1), which domain-separates
# the two and closes that attack.
#
# Manifests written before the hardening carry no algorithm field. Readers
# MUST treat a missing field as MERKLE_ALG_LEGACY: those roots were produced
# by the old construction, and defaulting to the new one reports an untampered
# artifact as a hash mismatch.
MERKLE_ALG_LEGACY = "sha256-concat"
MERKLE_ALG_RFC6962 = "rfc6962-sha256"
MERKLE_ALG_DEFAULT = MERKLE_ALG_RFC6962
MERKLE_ALGS = (MERKLE_ALG_LEGACY, MERKLE_ALG_RFC6962)


def _require_merkle_alg(alg: str) -> str:
    if alg not in MERKLE_ALGS:
        raise ValueError(f"unknown merkle_alg {alg!r}; expected one of {MERKLE_ALGS}")
    return alg


def _merkle_leaf(leaf_hash: bytes, alg: str = MERKLE_ALG_DEFAULT) -> bytes:
    if alg == MERKLE_ALG_LEGACY:
        return leaf_hash
    return compute_sha256_bytes(data=b"\x00" + leaf_hash)


def _merkle_parent(left: bytes, right: bytes, alg: str = MERKLE_ALG_DEFAULT) -> bytes:
    if alg == MERKLE_ALG_LEGACY:
        return compute_sha256_bytes(data=left + right)
    return compute_sha256_bytes(data=b"\x01" + left + right)


def merkle_root_from_leaf_hashes(
    leaf_hashes: List[str],
    *,
    alg: str = MERKLE_ALG_DEFAULT,
) -> str:
    _require_merkle_alg(alg)
    if not leaf_hashes:
        return compute_sha256(data=b"")

    level = [_merkle_leaf(bytes.fromhex(leaf), alg) for leaf in leaf_hashes]
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_merkle_parent(left, right, alg))
        level = next_level
    return level[0].hex()


def build_merkle_manifest(
    file_path: Union[str, Path],
    *,
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES,
    alg: str = MERKLE_ALG_DEFAULT,
) -> Dict[str, Any]:
    _require_merkle_alg(alg)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(file_path)
    chunks = []
    offset = 0
    file_hash = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
            chunks.append(
                {
                    "index": len(chunks),
                    "offset": offset,
                    "size": len(chunk),
                    "sha256": compute_sha256(data=chunk),
                }
            )
            offset += len(chunk)

    leaf_hashes = [chunk["sha256"] for chunk in chunks]
    return {
        "artifact": path.name,
        "size_bytes": offset,
        "sha256": file_hash.hexdigest(),
        "chunk_size_bytes": chunk_size,
        "chunk_count": len(chunks),
        "merkle_alg": alg,
        "merkle_root": merkle_root_from_leaf_hashes(leaf_hashes, alg=alg),
        "chunks": chunks,
    }


def write_merkle_manifest(
    file_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES,
) -> Dict[str, Any]:
    manifest = build_merkle_manifest(file_path, chunk_size=chunk_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_merkle_proof(
    file_path: Union[str, Path],
    chunk_index: int,
    *,
    chunk_size: int = MERKLE_CHUNK_SIZE_BYTES,
    alg: str = MERKLE_ALG_DEFAULT,
) -> List[Dict[str, Any]]:
    _require_merkle_alg(alg)
    manifest = build_merkle_manifest(file_path, chunk_size=chunk_size, alg=alg)
    if manifest["chunk_count"] == 0:
        raise ValueError("Cannot generate a Merkle proof for an empty file")
    if chunk_index < 0 or chunk_index >= manifest["chunk_count"]:
        raise IndexError("chunk_index out of range")

    level = [_merkle_leaf(bytes.fromhex(chunk["sha256"]), alg) for chunk in manifest["chunks"]]
    proof = []
    index = chunk_index
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        sibling_index = index ^ 1
        proof.append(
            {
                "sibling_sha256": level[sibling_index].hex(),
                "sibling_position": "left" if sibling_index < index else "right",
            }
        )

        next_level = []
        for i in range(0, len(level), 2):
            next_level.append(_merkle_parent(level[i], level[i + 1], alg))
        index //= 2
        level = next_level
    return proof


def verify_merkle_proof(
    chunk_bytes: bytes,
    proof: List[Dict[str, Any]],
    expected_root: str,
    *,
    alg: str = MERKLE_ALG_DEFAULT,
) -> bool:
    try:
        _require_merkle_alg(alg)
        raw_hash = compute_sha256_bytes(data=chunk_bytes)
        current = _merkle_leaf(raw_hash, alg)
        expected = bytes.fromhex(expected_root)
    except (TypeError, ValueError):
        return False

    for step in proof:
        try:
            sibling = bytes.fromhex(step["sibling_sha256"])
            position = step["sibling_position"]
        except (KeyError, TypeError, ValueError):
            return False

        if len(sibling) != hashlib.sha256().digest_size:
            return False
        if position == "left":
            current = _merkle_parent(sibling, current, alg)
        elif position == "right":
            current = _merkle_parent(current, sibling, alg)
        else:
            return False

    return current == expected


def _stable_cpu_state_dict(model: "torch.nn.Module") -> Dict[str, "torch.Tensor"]:
    state = model.state_dict()
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(state.items(), key=lambda item: item[0])
    }


def save_model_safetensors(
    model: "torch.nn.Module",
    output_path: Union[str, Path] = CHECKPOINT_WEIGHTS_PATH,
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to write stable model artifacts. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(_stable_cpu_state_dict(model), str(output), metadata=metadata)
    return output


def load_model_safetensors(
    model: "torch.nn.Module",
    input_path: Union[str, Path] = CHECKPOINT_WEIGHTS_PATH,
    *,
    device: Optional["torch.device"] = None,
) -> "torch.nn.Module":
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Safetensors artifact not found: {input_file}")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to read stable model artifacts. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    state = load_file(str(input_file), device=str(device) if device is not None else "cpu")
    model.load_state_dict(state)
    return model
