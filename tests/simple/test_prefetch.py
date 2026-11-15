"""Loader lifecycle checks using real synthetic memmaps; no GPU or release replay."""
import contextlib
import importlib.util
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ["OVL_DEVICE"] = "cpu"
os.environ["OVL_TIMESTAMP"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
spec = importlib.util.spec_from_file_location("ovl_prefetch_tested", ROOT / "simple" / "ovl.py")
ovl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ovl)


@pytest.fixture(params=["fast", "reference"])
def stream(tmp_path, monkeypatch, request):
    if request.param == "reference":
        monkeypatch.setenv("OVL_LOADER", "reference")
    else:
        monkeypatch.delenv("OVL_LOADER", raising=False)
    recipe = {**ovl.RECIPE, "context": 4, "batch_size": 2, "boundary_every": 1,
              "model": {"vocab_size": 16, "embed_dim": 8, "num_heads": 2, "num_layers": 1,
                        "max_seq_len": 4, "dropout": 0, "attn_impl": "manual"}}
    monkeypatch.setattr(ovl, "RECIPE", recipe)
    phase = tmp_path / "wikipedia"
    phase.mkdir()
    n = 80 * recipe["batch_size"] * recipe["context"]
    np.full(n, 5, dtype="<u2").tofile(phase / "tokens.u16")
    np.ones(n, dtype="u1").tofile(phase / "mask.u8")
    (phase / "documents.jsonl").write_text(json.dumps({"tokens": n, "offset": 0, "target_start": 0,
                                                     "targets": n}) + "\n")
    (phase / "stream.json").write_text(json.dumps({"targets": n}))
    name = "stream_batches" if request.param == "reference" else "fast_batches"
    loader = getattr(ovl, name)
    queues, closed = [], threading.Event()
    real_queue = queue.Queue

    class ObservedQueue(real_queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.full_event = threading.Event()
            self.worker = None
            queues.append(self)

        def put(self, item, *args, **kwargs):
            self.worker = threading.current_thread()
            if self.full():
                self.full_event.set()
            return super().put(item, *args, **kwargs)

    def observed_loader(*args, **kwargs):
        try:
            yield from loader(*args, **kwargs)
        finally:
            closed.set()

    monkeypatch.setattr(ovl.queue, "Queue", ObservedQueue)
    monkeypatch.setattr(ovl, name, observed_loader)
    yield tmp_path, loader, name, queues, closed
    # Drain only for test isolation on the unfixed implementation after a failing assertion.
    for q in queues:
        deadline = time.monotonic() + 5
        while q.worker and q.worker.is_alive() and time.monotonic() < deadline:
            try:
                q.get(timeout=0.05)
            except queue.Empty:
                pass
        if q.worker:
            q.worker.join(timeout=1)


def test_repeated_close_with_full_queue_releases_worker_and_loader(stream):
    data, _, _, queues, closed = stream
    for _ in range(3):
        closed.clear()
        batches = ovl.prefetch(data, "wikipedia", [0, 0])
        next(batches)
        q = queues[-1]
        assert q.full_event.wait(5), "producer did not reach the bounded queue"
        batches.close()
        q.worker.join(timeout=0.5)
        assert not q.worker.is_alive()
        assert closed.is_set()


def test_exhaustion_preserves_every_tensor_and_position(stream):
    data, loader, _, queues, closed = stream
    with contextlib.closing(ovl.prefetch(data, "wikipedia", [0, 4])) as batches:
        for (expected, expected_pos), (actual, actual_pos) in zip(loader(data / "wikipedia", [0, 4]), batches, strict=True):
            assert actual_pos == expected_pos
            assert actual.keys() == expected.keys()
            for key in expected:
                assert torch.equal(actual[key], expected[key])
    assert closed.is_set()
    queues[-1].worker.join(timeout=0.5)
    assert not queues[-1].worker.is_alive()


@pytest.mark.parametrize("fail_at", [0, 66])
def test_loader_error_reaches_consumer_and_releases_worker(stream, monkeypatch, fail_at):
    data, _, name, queues, _ = stream
    closed = threading.Event()

    def failing_loader(*args):
        try:
            for i in range(fail_at):
                yield i
            raise RuntimeError("loader failed")
        finally:
            closed.set()

    monkeypatch.setattr(ovl, name, failing_loader)
    batches = ovl.prefetch(data, "wikipedia", [0, 0])
    with pytest.raises(RuntimeError, match="loader failed"):
        list(batches)
    queues[-1].worker.join(timeout=0.5)
    assert not queues[-1].worker.is_alive()
    assert closed.is_set()


@pytest.mark.parametrize("terminal_error", [False, True])
def test_close_cancels_full_queue_terminal_delivery(stream, monkeypatch, terminal_error):
    data, _, name, queues, _ = stream

    def terminal_loader(*args):
        yield from range(65)
        if terminal_error:
            raise RuntimeError("queued loader error")

    monkeypatch.setattr(ovl, name, terminal_loader)
    batches = ovl.prefetch(data, "wikipedia", [0, 0])
    next(batches)
    q = queues[-1]
    assert q.full_event.wait(5)
    batches.close()
    q.worker.join(timeout=0.5)
    assert not q.worker.is_alive()
