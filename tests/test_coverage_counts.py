"""Complete census and short-document forecast counterexamples."""
import copy

import numpy as np
import pytest

from test_pipeline import prepared
from ovl_pipeline.budget import forecast
from ovl_pipeline.canonical import EvidenceError, Merkle, canonical, inventory, write_json
from ovl_pipeline.coverage import schedule_counts
from ovl_pipeline.data import EOS, batches
from ovl_pipeline.fixture import recipe
from test_budget import example


def stream_with_masks(root, documents, phase="wikipedia"):
    root.mkdir()
    tree = Merkle();offset = targets = 0;lines = []
    for i, mask in enumerate(documents):
        row = {"identity": [str(i)], "offset": offset, "tokens": len(mask),
               "target_start": targets, "targets": sum(mask)}
        lines.append(canonical(row) + b"\n");tree.add(canonical(row))
        offset += len(mask);targets += sum(mask)
    (root / "documents.jsonl").write_bytes(b"".join(lines))
    (root / "tokens.u16").write_bytes(np.full(offset, EOS, dtype="<u2").tobytes())
    (root / "mask.u8").write_bytes(bytes(v for mask in documents for v in mask))
    stream = {"schema": "ovl.stream.v1", "phase": phase, "token_dtype": "uint16-le",
              "tokenizer_sha256": "1" * 64, "documents": len(documents), "tokens": offset,
              "targets": targets, "index_root": tree.root(), "window_policy": "per-document-overlap-one-v1",
              "files": inventory(root, ["tokens.u16", "mask.u8", "documents.jsonl"])}
    write_json(root / "stream.json", stream)


def update_forecast(counts):
    value = example();value["schema"] = "ovl.cost-forecast-input.v2"
    for name in value["phases"]:
        value["phases"][name] = {"updates": counts["updates"], "training_completed": 0, "replay_completed": 0,
            "measured_full_batch_updates": 100, "measured_ms": 600000, "warmup_excluded": True,
            "overhead_included": True, "measurement_sha256": "a" * 64, "schedule_sha256": "d" * 64,
            "recipe_sha256": counts["recipe_sha256"], "stream_sha256": counts["stream_sha256"]}
    return value


def test_identical_target_count_can_require_sixteen_times_more_compute(tmp_path):
    r = recipe(320)
    stream_with_masks(tmp_path / "long", [[1] * 96])
    stream_with_masks(tmp_path / "short", [[1]] * 96)
    long = schedule_counts(tmp_path / "long", r);short = schedule_counts(tmp_path / "short", r)
    assert long["targets"] == short["targets"] == 96
    assert long["updates"] == 2 and short["updates"] == 32
    assert long["padded_positions"] == 0 and short["padded_positions"] == 1440
    costs = [forecast(update_forecast(s))["remaining_compute_micro_usd"] for s in (long, short)]
    assert costs[1] == 16 * costs[0]


def test_mask_only_windows_and_partial_batch_counted_exactly(tmp_path):
    r = recipe(320)
    # Two all-user windows disappear; two full assistant windows plus a tail remain.
    stream_with_masks(tmp_path / "chat", [[0] * 32 + [1] * 33, [0] * 17 + [1]], "conversation")
    c = schedule_counts(tmp_path / "chat", r)
    assert (c["target_bearing_windows"], c["updates"], c["final_batch_rows"], c["targets"]) == (4, 2, 1, 34)
    assert c["padded_positions"] == 29 and c["masked_context_positions"] == 1
    (tmp_path / "chat/mask.u8").write_bytes(b"\0" * 83)
    with pytest.raises(EvidenceError):schedule_counts(tmp_path / "chat", r)


@pytest.mark.parametrize("phase", ["wikipedia", "conversation"])
def test_independent_census_matches_actual_complete_loader(prepared, phase, monkeypatch):
    root, manifest = prepared;r = recipe(manifest["tokenizer"]["vocab_size"])
    actual = list(batches(root / phase, r["context"], r["batch_size"]))
    monkeypatch.setattr("ovl_pipeline.data.batches", lambda *a: pytest.fail("census must not call loader"))
    c = schedule_counts(root / phase, r)
    assert c["updates"] == len(actual)
    assert c["target_bearing_windows"] == sum(b["inputs"].shape[0] for b in actual)
    assert c["targets"] == sum(int(b["mask"].sum()) for b in actual)


def test_update_forecast_refuses_missing_census_or_zero_full_batch_measurement(tmp_path):
    stream_with_masks(tmp_path / "stream", [[1] * 96])
    value = update_forecast(schedule_counts(tmp_path / "stream", recipe(320)))
    for field, new in (("schedule_sha256", ""), ("measured_full_batch_updates", 0), ("replay_completed", 1)):
        changed = copy.deepcopy(value);changed["phases"]["wikipedia"][field] = new
        with pytest.raises(EvidenceError):forecast(changed)
    result = forecast(value)
    assert result["schema"] == "ovl.cost-forecast.v2"
    assert result["phases"]["wikipedia"]["remaining_updates_including_replay"] == 4
