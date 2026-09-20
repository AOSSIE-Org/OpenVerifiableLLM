"""Safe complete-state capture for the declared single-process training kernel."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import struct
import sys

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from .canonical import EvidenceError, canonical, confined, digest, inventory, read_json, verify_inventory, write_json


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    if sys.byteorder != "little":
        raise EvidenceError("only little-endian hosts supported")
    h = hashlib.sha256(b"ovl.tensors.v1\x00")
    h.update(struct.pack(">Q", len(tensors)))
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().cpu().contiguous()
        if t.layout != torch.strided or t.is_quantized:
            raise EvidenceError("unsupported tensor layout")
        header = canonical({"name": name, "dtype": str(t.dtype), "shape": list(t.shape), "endian": "little"})
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, raw):
            h.update(struct.pack(">Q", len(part)))
            h.update(part)
    return h.hexdigest()


def pack(value, tensors, path="root"):
    """Tagged encoding preserves Python type, finite float bits and tensor identity."""
    if isinstance(value, torch.Tensor):
        if path in tensors:
            raise EvidenceError("duplicate state tensor path")
        tensors[path] = value.detach().cpu().contiguous().clone()
        return {"type": "tensor", "name": path}
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError("nonfinite state scalar")
        return {"type": "float64", "hex": struct.pack(">d", value).hex()}
    if type(value) in (list, tuple):
        return {"type": type(value).__name__, "items": [pack(v, tensors, f"{path}/{i}") for i, v in enumerate(value)]}
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise EvidenceError("state dictionary keys must be strings")
        # Index-based storage paths avoid collisions in escaped parameter names.
        return {"type": "dict", "items": [[k, pack(v, tensors, f"{path}/{i}")] for i, (k, v) in enumerate(sorted(value.items()))]}
    if value is None or type(value) in (str, int, bool):
        canonical(value)
        return {"type": "scalar", "value": value}
    raise EvidenceError(f"unsupported state type: {type(value)}")


def unpack(value, tensors):
    tag = value["type"]
    if tag == "tensor" and set(value) == {"type", "name"}:
        return tensors[value["name"]].clone()
    if tag == "float64" and set(value) == {"type", "hex"}:
        result = struct.unpack(">d", bytes.fromhex(value["hex"]))[0]
        if not math.isfinite(result):
            raise EvidenceError("nonfinite scalar")
        return result
    if tag == "scalar" and set(value) == {"type", "value"}:
        return value["value"]
    if tag in ("list", "tuple") and set(value) == {"type", "items"}:
        values = [unpack(v, tensors) for v in value["items"]]
        return tuple(values) if tag == "tuple" else values
    if tag == "dict" and set(value) == {"type", "items"}:
        items = value["items"]
        if len({k for k, _ in items}) != len(items):
            raise EvidenceError("duplicate packed key")
        return {k: unpack(v, tensors) for k, v in items}
    raise EvidenceError("unknown packed state")


def aliases(model):
    seen, ties = {}, {}
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            ties[name] = seen[id(p)]
        else:
            seen[id(p)] = name
    return ties


def capture(model, optimizer, control):
    if any(p.grad is not None for p in model.parameters()):
        raise EvidenceError("checkpoints require cleared gradients at an update boundary")
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append({**{k: v for k, v in group.items() if k != "params"},
                       "params": [names[id(p)] for p in group["params"]]})
    np_rng = np.random.get_state()
    obj = {
        "model": dict(model.state_dict()),
        "aliases": aliases(model),
        "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
        "optimizer": {names[id(p)]: s for p, s in optimizer.state.items()},
        "groups": groups,
        "modes": {n: m.training for n, m in model.named_modules()},
        "control": control,
        "rng": {"python": random.getstate(), "numpy": [np_rng[0], np_rng[1].tolist(), *np_rng[2:]],
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []},
    }
    tensors = {}
    metadata = {"schema": "ovl.state.v1", "tree": pack(obj, tensors)}
    metadata["tensor_root"] = tensor_digest(tensors)
    return metadata, tensors


def state_root(metadata, tensors):
    if set(metadata) != {"schema", "tree", "tensor_root"} or metadata.get("schema") != "ovl.state.v1" or metadata.get("tensor_root") != tensor_digest(tensors):
        raise EvidenceError("state tensor root mismatch")
    return digest(metadata)


def restore(model, optimizer, metadata, tensors):
    state_root(metadata, tensors)
    obj = unpack(metadata["tree"], tensors)
    if aliases(model) != obj["aliases"]:
        raise EvidenceError("model alias mismatch")
    if obj["optimizer_class"] != type(optimizer).__module__ + "." + type(optimizer).__qualname__:
        raise EvidenceError("optimizer class mismatch")
    for alias, primary in obj["aliases"].items():
        if tensor_digest({"tensor": obj["model"][alias]}) != tensor_digest({"tensor": obj["model"][primary]}):
            raise EvidenceError("inconsistent tied tensor payload")
    model.load_state_dict(obj["model"], strict=True)
    names = dict(model.named_parameters())
    if len(obj["groups"]) != len(optimizer.param_groups):
        raise EvidenceError("optimizer group count mismatch")
    for existing, saved in zip(optimizer.param_groups, obj["groups"]):
        if [id(p) for p in existing["params"]] != [id(names[n]) for n in saved["params"]]:
            raise EvidenceError("optimizer parameter order mismatch")
        existing.clear()
        existing.update({**saved, "params": [names[n] for n in saved["params"]]})
    optimizer.state.clear()
    for name, values in obj["optimizer"].items():
        p = names[name]
        # This kernel only supports noncapturable, nonfused AdamW. Its step is CPU.
        optimizer.state[p] = {k: (v.to(p.device) if isinstance(v, torch.Tensor) and k != "step" else v)
                              for k, v in values.items()}
    for n, module in model.named_modules():
        module.training = obj["modes"][n]
    random.setstate(obj["rng"]["python"])
    nr = obj["rng"]["numpy"]
    np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
    torch.set_rng_state(obj["rng"]["torch_cpu"])
    if obj["rng"]["torch_cuda"]:
        if len(obj["rng"]["torch_cuda"]) != torch.cuda.device_count():
            raise EvidenceError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(obj["rng"]["torch_cuda"])
    optimizer.zero_grad(set_to_none=True)
    if state_root(*capture(model, optimizer, obj["control"])) != state_root(metadata, tensors):
        raise EvidenceError("restored state differs from checkpoint")
    return obj["control"]


def save_state(directory: Path, model, optimizer, control):
    directory.mkdir(parents=True, exist_ok=False)
    metadata, tensors = capture(model, optimizer, control)
    save_file(tensors, str(directory / "state.safetensors"))
    # Flush tensor file before publishing the completion manifest.
    with (directory / "state.safetensors").open("rb") as f:
        import os
        os.fsync(f.fileno())
    write_json(directory / "state.json", metadata)
    manifest = {"schema": "ovl.checkpoint.v1", "state_root": state_root(metadata, tensors),
                "files": inventory(directory, ["state.json", "state.safetensors"])}
    write_json(directory / "checkpoint.json", manifest)
    return manifest


def read_state(directory: Path, expected_manifest):
    if set(expected_manifest) != {"schema", "state_root", "files"} or expected_manifest.get("schema") != "ovl.checkpoint.v1":
        raise EvidenceError("unknown checkpoint schema")
    if {e["path"] for e in expected_manifest["files"]} != {"state.json", "state.safetensors"}:
        raise EvidenceError("unexpected checkpoint inventory")
    marker = confined(directory, "checkpoint.json")
    if not marker.is_file() or read_json(marker) != expected_manifest:
        raise EvidenceError("missing or mismatched checkpoint completion manifest")
    verify_inventory(directory, expected_manifest["files"], max_bytes=2 * 1024**3)
    metadata = read_json(directory / "state.json")
    tensors = load_file(str(directory / "state.safetensors"))
    if state_root(metadata, tensors) != expected_manifest["state_root"]:
        raise EvidenceError("checkpoint state mismatch")
    return metadata, tensors
