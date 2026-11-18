"""Segment-slice loader equivalence tests using synthetic prepared data."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))

spec = importlib.util.spec_from_file_location(
    "ovl_segment_slice_tested", ROOT / "simple" / "ovl.py"
)
ovl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ovl)


@pytest.fixture
def prepared_stream(tmp_path, monkeypatch):
    recipe = {
        **ovl.RECIPE,
        "context": 4,
        "batch_size": 3,
    }
    monkeypatch.setattr(ovl, "RECIPE", recipe)

    phase = tmp_path / "wikipedia"
    phase.mkdir()

    documents = [
        (7,  [1, 1, 0, 1, 1, 0, 1]),
        (5,  [1, 0, 1, 0, 1]),
        (10, [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]),
        (3,  [1, 1, 0]),
        (2,  [1, 0]),
        (4,  [1, 1, 1, 1]),
    ]

    token_parts = []
    mask_parts = []
    rows = []
    row_offsets = []
    token_offset = 0
    target_cursor = 0
    json_offset = 0

    for index, (length, mask_values) in enumerate(documents):
        tokens = np.full(length, 10 + index, dtype="<u2")
        mask = np.asarray(mask_values, dtype=np.uint8)

        row = {
            "identity": ["synthetic", str(index)],
            "offset": token_offset,
            "tokens": length,
            "target_start": target_cursor,
            "targets": int(mask.sum()),
        }

        encoded = (json.dumps(row, separators=(",", ":")) + "\n").encode()
        row_offsets.append(json_offset)
        json_offset += len(encoded)

        rows.append(encoded)
        token_parts.append(tokens)
        mask_parts.append(mask)

        token_offset += length
        target_cursor += int(mask.sum())

    (phase / "tokens.u16").write_bytes(
        np.concatenate(token_parts).astype("<u2").tobytes()
    )
    (phase / "mask.u8").write_bytes(
        np.concatenate(mask_parts).astype(np.uint8).tobytes()
    )

    documents_path = phase / "documents.jsonl"
    documents_path.write_bytes(b"".join(rows))

    (phase / "stream.json").write_text(
        json.dumps({"targets": target_cursor}),
        encoding="utf-8",
    )

    # Exclude document 1 using its original global target_start.
    skip = frozenset({3 + 2})  # document 1 starts after document 0's 5 targets

    return phase, row_offsets, documents_path.stat().st_size, skip


def assert_batches_equal(left, right):
    assert len(left) == len(right)

    for (left_batch, left_pos), (right_batch, right_pos) in zip(
        left, right, strict=True
    ):
        assert left_pos == right_pos
        assert left_batch.keys() == right_batch.keys()

        for key in left_batch:
            assert torch.equal(left_batch[key], right_batch[key]), key


@pytest.mark.parametrize("loader_name", ["stream_batches", "fast_batches"])
def test_segment_resume_preserves_exact_batches_and_positions(
    prepared_stream, loader_name
):
    phase, row_offsets, file_size, skip = prepared_stream
    loader = getattr(ovl, loader_name)

    full = list(loader(phase, [0, 0], skip))

    # Seven nonempty windows produce two full batches and one partial batch.
    assert len(full) == 3

    # The first batch crosses from document 0 into document 2.
    assert full[0][0]["targets"][:, 0].tolist() == [10, 10, 12]
    assert full[0][1] == [row_offsets[2], 4]

    # The next batch crosses two more document boundaries.
    assert full[1][0]["targets"][:, 0].tolist() == [12, 13, 14]
    assert full[1][1] == [row_offsets[5], 0]

    # The final batch is partial: only its first row contains a window.
    assert full[2][0]["mask"].sum(dim=1).tolist() == [4]
    assert full[2][0]["mask"].shape == (1, 4)
    assert full[2][1] == [file_size, 0]

    # Original global IDs are retained; only the excluded document's
    # target range [5, 8) is absent.
    observed_ids = torch.cat(
        [batch["target_ids"][batch["mask"]] for batch, _ in full]
    ).tolist()

    expected_ids = list(range(5)) + list(range(8, 19))
    assert observed_ids == expected_ids

    # Resume from the first batch's exact recorded position.
    resumed = list(loader(phase, full[0][1], skip))

    assert len(resumed) == 2
    assert_batches_equal(resumed, full[1:])

    # The resumed segment ends at exactly the same recorded position.
    assert resumed[-1][1] == full[-1][1]


def test_fast_loader_matches_reference_for_entire_stream(prepared_stream):
    phase, _, _, skip = prepared_stream

    reference = list(ovl.stream_batches(phase, [0, 0], skip))
    optimized = list(ovl.fast_batches(phase, [0, 0], skip))

    assert_batches_equal(reference, optimized)