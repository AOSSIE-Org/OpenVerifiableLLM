"""Build and independently verify a saved OVL segment-slice prototype.

Synthetic-data research only. This does not alter production verification and
does not cryptographically authenticate the package manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import ovl

ovl.DEV = "cpu"
ovl.RECIPE = {**ovl.RECIPE, "context": 4, "batch_size": 2}
HERE = Path(__file__).resolve().parent
DATA = HERE / "synthetic_data" / "wikipedia"
PACKAGE_DIR = HERE / "saved_packages"
CONTEXT = ovl.RECIPE["context"]
BATCH_SIZE = ovl.RECIPE["batch_size"]


def batch_digest(batch):
    h = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8") + b"\\0")
        h.update(str(value.dtype).encode("ascii") + b"\\0")
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\\0")
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def extract_windows(start_pos, end_pos):
    tokens = np.memmap(DATA / "tokens.u16", dtype="<u2", mode="r")
    mask = np.memmap(DATA / "mask.u8", dtype="u1", mode="r")
    windows = []
    with (DATA / "documents.jsonl").open("rb") as f:
        f.seek(start_pos[0])
        first = start_pos[1]
        while True:
            row = f.tell()
            if row > end_pos[0] or (row == end_pos[0] and end_pos[1] == 0):
                break
            line = f.readline()
            if not line:
                break
            d = json.loads(line)
            for start in range(0, d["tokens"], CONTEXT):
                if row == start_pos[0] and start < first:
                    continue
                if row == end_pos[0] and start >= end_pos[1]:
                    break
                n = min(CONTEXT, d["tokens"] - start)
                absolute = d["offset"] + start
                valid = mask[absolute:absolute + n].astype(bool)
                if not int(valid.sum()):
                    continue
                y = tokens[absolute:absolute + n].astype(np.int64).tolist()
                base = d["target_start"] + int(mask[d["offset"]:absolute].sum())
                previous = int(ovl.BOS) if start == 0 else int(tokens[absolute - 1])
                after = ([row, start + CONTEXT] if start + CONTEXT < d["tokens"]
                         else [f.tell(), 0])
                windows.append({
                    "document_row": row, "window_start": start, "tokens": y,
                    "mask": valid.astype(int).tolist(), "target_start": base,
                    "previous_token": previous, "after": after,
                })
            first = 0
    return windows


def reconstruct(windows):
    rows = []
    for item in windows:
        y = np.asarray(item["tokens"], dtype=np.int64)
        valid = np.asarray(item["mask"], dtype=bool)
        ids = np.full(len(y), -1, dtype=np.int64)
        ids[valid] = np.arange(item["target_start"],
                               item["target_start"] + int(valid.sum()))
        x = np.concatenate((np.asarray([item["previous_token"]]), y[:-1]))
        rows.append((x, y, valid, ids))
    batches = []
    for i in range(0, len(rows), BATCH_SIZE):
        group = rows[i:i + BATCH_SIZE]
        last = windows[i + len(group) - 1]
        batches.append((ovl._collate(group, CONTEXT), last["after"]))
    return batches


def build():
    full = list(ovl.stream_batches(DATA, [0, 0]))
    experiments = [(full[0][1], full[1][1]), (full[1][1], full[2][1])]
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    for number, (start, end) in enumerate(experiments, 1):
        expected = []
        for batch in ovl.stream_batches(DATA, start):
            expected.append(batch)
            if batch[1] == end:
                break
        if not expected or expected[-1][1] != end:
            raise AssertionError("full loader did not reach requested segment end")
        windows = extract_windows(start, end)
        actual = reconstruct(windows)
        if len(actual) != len(expected):
            raise AssertionError("reconstructed batch count differs")
        for i, ((got, got_pos), (want, want_pos)) in enumerate(zip(actual, expected)):
            if got_pos != want_pos:
                raise AssertionError(f"ending position mismatch in batch {i}")
            for key in want:
                if not torch.equal(got[key], want[key]):
                    raise AssertionError(f"tensor mismatch in batch {i}: {key}")
        package = {
            "format": "ovl-segment-slice-prototype.v1",
            "synthetic_only": True,
            "context": CONTEXT,
            "batch_size": BATCH_SIZE,
            "start_position": start,
            "end_position": end,
            "windows": windows,
            "expected_batches": [
                {"sha256": batch_digest(batch), "after": pos}
                for batch, pos in expected
            ],
        }
        path = PACKAGE_DIR / f"segment_{number}.json"
        path.write_text(json.dumps(package, separators=(",", ":"), sort_keys=True),
                        encoding="utf-8")
        print(f"BUILT {path.name}: {len(windows)} windows, {len(expected)} batches")
        print("  Full-loader tensor and position equivalence: PASS")
        print(f"  Package bytes: {path.stat().st_size}")


def verify():
    paths = sorted(PACKAGE_DIR.glob("segment_*.json"))
    if not paths:
        raise FileNotFoundError(f"No packages in {PACKAGE_DIR}; run build first")
    for path in paths:
        package = json.loads(path.read_text(encoding="utf-8"))
        if package.get("format") != "ovl-segment-slice-prototype.v1":
            raise ValueError(f"{path.name}: unsupported format")
        if package.get("synthetic_only") is not True:
            raise ValueError(f"{path.name}: expected synthetic-only marker")
        if package.get("context") != CONTEXT or package.get("batch_size") != BATCH_SIZE:
            raise ValueError(f"{path.name}: recipe mismatch")
        batches = reconstruct(package["windows"])
        expected = package["expected_batches"]
        if len(batches) != len(expected):
            raise AssertionError(f"{path.name}: batch count mismatch")
        for i, ((batch, after), item) in enumerate(zip(batches, expected)):
            if after != item["after"]:
                raise AssertionError(f"{path.name}: position mismatch in batch {i}")
            if batch_digest(batch) != item["sha256"]:
                raise AssertionError(f"{path.name}: digest mismatch in batch {i}")
        if batches and batches[-1][1] != package["end_position"]:
            raise AssertionError(f"{path.name}: final position mismatch")
        print(f"VERIFIED {path.name}: {len(batches)} batches; digest and positions PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    mode = parser.parse_args().mode
    build() if mode == "build" else verify()
