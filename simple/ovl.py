"""OpenVerifiableLLM, simple production loop: train -> verify segments -> chat.

One file, one GPU. Reuses the project's prepared streams, batch order and model:
  G03 reproducible init     seeded CPU init (no pretrained weights); `verify` regenerates it
  G04 exactly one pass      every target id checked in order; Wikipedia covered exactly once
  G05 committed progress    chain.jsonl: hash-linked state digests at fixed boundaries, and
                            every boundary checkpoint kept so anyone can start from it
  G06 conversation phase    same loop on the conversation stream, fresh AdamW at transition
  G07 exact replay          split into segments (boundary k -> k+1). Each segment is checked
                            independently, bit for bit, so verification runs in parallel and
                            the model is usable as soon as training ends.

Usage (on a pod):
  python simple/ovl.py train  --data PREP --out RUN
  python simple/ovl.py verify --data PREP --run RUN --random 2        # spot check
  python simple/ovl.py verify --data PREP --run RUN --segment 7 8     # chosen segments
  python simple/ovl.py verify --data PREP --run RUN --all             # full replay
  python simple/ovl.py verify --run RUN --weights RUN/conversation-model.pt
  python simple/ovl.py chat   --run RUN --tokenizer PREP/tokenizer/tokenizer.json "What is Wikipedia?"
  python simple/ovl.py bench  --data PREP --seconds 600

Each new chain entry is also timestamped by two RFC 3161 authorities while training runs
(simple/stamp.py; OVL_TIMESTAMP=0 turns it off), so the record shows when each boundary existed.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import math
from pathlib import Path
import queue
import secrets
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch
import torch.nn.functional as F

from ovl_pipeline.data import PAD, BOS, EOS, USER, ASSISTANT, OFFSET, _collate, stream_arrays
from ovl_pipeline import training
from model import TinyGPT
import stamp

RECIPE = {"batch_size": 32, "boundary_every": 18080, "context": 512, "init": "normal-0.02-tied-v1",
          "seed": 1234, "weight_decay": "0.01", "conversation_epochs": 3,
          # Learning rate is a pure function of the step, so a segment started from any
          # checkpoint uses exactly the rates the original run used.
          "schedule": {"peak": "0.0006", "floor": "0.00006", "warmup": 2000, "wikipedia_updates": 560177,
                       "conversation": "0.0001"},
          "model": {"attn_impl": "manual", "dropout": 0, "embed_dim": 384, "max_seq_len": 512,
                    "num_heads": 6, "num_layers": 6, "vocab_size": 32000}}
DEV = os.environ.get("OVL_DEVICE", "cuda")  # "cpu" only for smoke tests
if os.environ.get("OVL_BOUNDARY_EVERY"):  # smoke tests only
    RECIPE["boundary_every"] = int(os.environ["OVL_BOUNDARY_EVERY"])
    RECIPE["schedule"].update(warmup=2, wikipedia_updates=4)
STOP = "stop"
TIMESTAMP = os.environ.get("OVL_TIMESTAMP", "1") != "0"


def stages():
    return [("wikipedia", 0)] + [("conversation", e) for e in range(RECIPE["conversation_epochs"])]


def deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def cpu_threads():
    """CPU threads for the loader after the single-threaded init. GPU pods are CPU-quota
    limited (cgroup cpu.max) but see every host core; a pool sized to all of them gets the
    whole container throttled and starves the GPU. Only integer/indexing work and byte
    copies run on the CPU after init, so this never changes a computed bit."""
    if os.environ.get("OVL_THREADS"):
        return int(os.environ["OVL_THREADS"])
    n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return max(1, min(n, 4))


def environment():
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if DEV == "cuda" else "cpu",
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def learning_rate(ctl):
    s = RECIPE["schedule"]
    if ctl["phase"] == "conversation":
        return float(s["conversation"])
    peak, floor, warm, n = float(s["peak"]), float(s["floor"]), s["warmup"], s["wikipedia_updates"]
    step = ctl["global_step"]  # index of the update about to run
    if step < warm:
        return peak * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, n - warm))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=float(RECIPE["schedule"]["peak"]),
                             weight_decay=float(RECIPE["weight_decay"]), betas=(0.9, 0.999), eps=1e-8,
                             foreach=True, fused=False)


def fresh():
    """Seeded CPU init exactly as the project defines it, then moved to the GPU."""
    deterministic()
    # Same arithmetic as ovl_pipeline.training.initialize, minus its CPU-size gate.
    training.configure(RECIPE["seed"])
    model = TinyGPT(**RECIPE["model"])
    model.lm_head.weight = model.transformer["wte"].weight
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        elif name.endswith("weight"):
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.zeros_(p)
    model.train()
    torch.set_num_threads(cpu_threads())
    model.to(DEV)
    model.lm_head.weight = model.transformer["wte"].weight
    return model, optimizer(model)


def _digest(tensors):
    h = hashlib.sha256()
    for name, t in tensors:
        t = t.detach().reshape(-1).contiguous().cpu()
        h.update(f"{name}|{t.dtype}|{t.numel()}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def model_digest(state_dict):
    return _digest(sorted(state_dict.items()))


def state_digest(model, opt):
    tensors = [(f"model.{k}", v) for k, v in model.state_dict().items()]
    for i, p in enumerate(model.parameters()):
        for k, v in sorted(opt.state.get(p, {}).items()):
            tensors.append((f"opt.{i}.{k}", v if torch.is_tensor(v) else torch.tensor(v)))
    return _digest(tensors)


def stream_batches(directory, pos, skip=frozenset()):
    """ovl_pipeline.data.batches, but able to start at a recorded position.

    pos = [byte offset of a documents.jsonl row, first window start in that row]. Each
    yielded batch comes with the position just after it. The order and contents are
    identical to batches(); the running batch hash in the chain catches any difference.
    """
    context, batch_size = RECIPE["context"], RECIPE["batch_size"]
    tok, mask = stream_arrays(directory)
    path = directory / "documents.jsonl"
    end = path.stat().st_size
    pending = []
    with path.open("rb") as f:
        f.seek(pos[0])
        first = pos[1]
        while True:
            row = f.tell()
            line = f.readline()
            if not line:
                break
            d = json.loads(line)
            if d["target_start"] in skip:  # removed by a recorded fork
                first = 0
                continue
            target = d["target_start"]
            for start in range(0, d["tokens"], context):
                n = min(context, d["tokens"] - start)
                absolute = d["offset"] + start
                valid = torch.from_numpy(mask[absolute:absolute + n].astype(bool))
                count = int(valid.sum())
                if start < first:  # consumed before the recorded position
                    target += count
                    continue
                y = torch.from_numpy(tok[absolute:absolute + n].astype("int64"))
                ids = torch.full((n,), -1, dtype=torch.int64)
                ids[valid] = torch.arange(target, target + count)
                target += count
                if not count:
                    continue
                previous = BOS if start == 0 else int(tok[absolute - 1])
                x = torch.cat((torch.tensor([previous]), y[:-1]))
                pending.append((x.numpy(), y.numpy(), valid.numpy(), ids.numpy()))
                if len(pending) == batch_size:
                    after = [row, start + context] if start + context < d["tokens"] else [f.tell(), 0]
                    yield _collate(pending, context), after
                    pending = []
            first = 0
    if pending:
        yield _collate(pending, context), [end, 0]


def fast_batches(directory, pos, skip=frozenset()):
    """stream_batches with the per-window work vectorized: one mask read and reduceat per
    document, one gather per batch, pinned host memory. Same batches, same positions; the
    running batch hash in the chain and segment verification catch any difference."""
    import numpy as np
    context, batch_size = RECIPE["context"], RECIPE["batch_size"]
    tok, mask = stream_arrays(directory)
    path = directory / "documents.jsonl"
    end = path.stat().st_size
    cols = np.arange(context)
    pending = []  # (absolute, length, first target id, previous token)

    def collate(w):
        a, n, base, prev = (np.array(c, dtype=np.int64) for c in zip(*w))
        inside = cols < n[:, None]
        idx = np.where(inside, a[:, None] + cols, a[:, None])
        y = np.where(inside, tok[idx].astype(np.int64), PAD)
        valid = inside & (mask[idx] != 0)
        x = np.empty_like(y)
        x[:, 0] = prev
        x[:, 1:] = y[:, :-1]
        x = np.where(inside, x, PAD)
        ids = np.where(valid, base[:, None] + np.cumsum(valid, axis=1) - 1, -1)
        out = {"inputs": x, "targets": y, "mask": valid, "target_ids": ids}
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
        return {k: v.pin_memory() for k, v in out.items()} if DEV == "cuda" else out

    with path.open("rb") as f:
        f.seek(pos[0])
        first = pos[1]
        while True:
            row = f.tell()
            line = f.readline()
            if not line:
                break
            d = json.loads(line)
            n_tok, offset = d["tokens"], d["offset"]
            if not n_tok or d["target_start"] in skip:  # empty, or removed by a recorded fork
                first = 0
                continue
            starts = np.arange(0, n_tok, context)
            counts = np.add.reduceat(mask[offset:offset + n_tok] != 0, starts, dtype=np.int64)
            bases = d["target_start"] + np.concatenate(([0], np.cumsum(counts)[:-1]))
            for start, count, base in zip(starts.tolist(), counts.tolist(), bases.tolist()):
                if start < first or not count:  # consumed before the recorded position, or no targets
                    continue
                absolute = offset + start
                previous = BOS if start == 0 else int(tok[absolute - 1])
                pending.append((absolute, min(context, n_tok - start), base, previous))
                if len(pending) == batch_size:
                    after = [row, start + context] if start + context < n_tok else [f.tell(), 0]
                    yield collate(pending), after
                    pending = []
            first = 0
    if pending:
        yield collate(pending), [end, 0]


def prefetch(data, phase, pos, skip=frozenset()):
    """Batches on a background thread (vectorized loader; OVL_LOADER=reference for the original)."""
    q = queue.Queue(maxsize=64)
    load = stream_batches if os.environ.get("OVL_LOADER") == "reference" else fast_batches

    def work():
        try:
            for item in load(Path(data) / phase, pos, skip):
                q.put(item)
        except BaseException as e:  # surface loader errors in the main thread
            q.put(e)
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def excluded(ctl):
    """Target-id ranges [start, count] that a recorded fork removed from this phase."""
    return sorted(tuple(r) for r in ctl.get("exclude", {}).get(ctl["phase"], []))


def next_target(cursor, ranges):
    """The first target id at or after cursor that is not in an excluded range."""
    for s, n in ranges:
        if s <= cursor < s + n:
            cursor = s + n
    return cursor


def step(model, opt, b, ctl):
    ids = b["target_ids"][b["mask"]]
    ranges = excluded(ctl)
    first = next_target(ctl["cursor"], ranges)
    last = first + ids.numel() - 1
    for s, n in ranges:  # an excluded document inside the batch's span widens it by exactly its size
        if first < s <= last:
            last += n
    if ids.numel() == 0 or ids[0].item() != first or ids[-1].item() != last:
        raise SystemExit(f"coverage error at cursor {ctl['cursor']}")  # gap, repeat or reorder
    for g in opt.param_groups:
        g["lr"] = learning_rate(ctl)
    x, y, m = (b[k].to(DEV, non_blocking=True) for k in ("inputs", "targets", "mask"))
    with torch.autocast(DEV, dtype=torch.bfloat16, cache_enabled=False):
        logits = model(x)
        losses = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction="none")
        m = m.flatten()
        loss = losses[m].sum() / m.sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss, last + 1, hashlib.sha256(b["inputs"].numpy().tobytes()).digest()


def targets(data, phase):
    return json.loads((Path(data) / phase / "stream.json").read_text())["targets"]


def initial_control():
    return {"phase": "wikipedia", "epoch": 0, "global_step": 0, "phase_step": 0, "cursor": 0, "pos": [0, 0],
            "batches": hashlib.sha256(b"ovl.simple.batches.v2").hexdigest()}


def run_phases(data, model, opt, on_boundary, start=None):
    """Drive every stage from `start`; call on_boundary(kind, control, opt) at each boundary.
    on_boundary may return STOP to end early (segment verification)."""
    ctl = start or initial_control()
    if start is None and on_boundary("initial", ctl, opt) == STOP:
        return ctl
    plan = stages()
    for phase, epoch in plan[plan.index((ctl["phase"], ctl["epoch"])):]:
        if (phase, epoch) != (ctl["phase"], ctl["epoch"]):
            if phase != ctl["phase"]:
                opt = optimizer(model)
            ctl = {**ctl, "phase": phase, "epoch": epoch, "phase_step": 0, "cursor": 0, "pos": [0, 0]}
            if on_boundary("transition", ctl, opt) == STOP:
                return ctl
        total, t0, loss = targets(data, phase), time.time(), None
        ranges = excluded(ctl)
        for b, after in prefetch(data, phase, ctl["pos"], frozenset(s for s, _ in ranges)):
            loss, cursor, bh = step(model, opt, b, ctl)
            ctl = {**ctl, "global_step": ctl["global_step"] + 1, "phase_step": ctl["phase_step"] + 1,
                   "cursor": cursor, "pos": after,
                   "batches": hashlib.sha256(bytes.fromhex(ctl["batches"]) + bh).hexdigest()}
            if ctl["phase_step"] % 1000 == 0:
                lv = loss.item()
                if not math.isfinite(lv):
                    raise SystemExit(f"nonfinite loss at step {ctl['global_step']}")
                rate = ctl["phase_step"] / (time.time() - t0)
                print(f"{phase} step {ctl['phase_step']} loss {lv:.4f} {1000 / rate:.1f} ms/update", flush=True)
            end = next_target(ctl["cursor"], ranges) == total
            if end or ctl["phase_step"] % RECIPE["boundary_every"] == 0:
                kind = ("final" if (phase, epoch) == plan[-1] else "base" if phase == "wikipedia" else "epoch") \
                    if end else "progress"
                if on_boundary(kind, ctl, opt) == STOP:
                    return ctl
        if next_target(ctl["cursor"], ranges) != total:
            raise SystemExit(f"{phase}: covered {ctl['cursor']} of {total} targets")
    return ctl


def read_chain(run):
    chain = [json.loads(l) for l in (Path(run) / "chain.jsonl").read_text().splitlines()]
    for i, e in enumerate(chain):
        want = hashlib.sha256(json.dumps(chain[i - 1], sort_keys=True).encode()).hexdigest() if i else None
        if e["index"] != i or e["previous"] != want:
            sys.exit(f"chain link broken at entry {i}")
    return chain


def load_checkpoint(run, entry, model):
    """Load boundary checkpoint `entry` and prove it is the state the chain committed to."""
    ck = torch.load(Path(run) / f"ckpt-{entry['index']:05d}.pt", map_location=DEV, weights_only=True)
    if ck["control"] != entry["control"]:
        sys.exit(f"checkpoint {entry['index']} control differs from the chain")
    model.load_state_dict(ck["model"])
    model.lm_head.weight = model.transformer["wte"].weight
    opt = optimizer(model)
    opt.load_state_dict(ck["opt"])
    if state_digest(model, opt) != entry["state"]:
        sys.exit(f"checkpoint {entry['index']} state differs from the chain")
    return opt


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chain = read_chain(out) if (out / "chain.jsonl").exists() else []
    model, opt = fresh()
    start = None
    if chain:
        if chain[-1]["kind"] == "final":
            sys.exit("run already complete")
        opt = load_checkpoint(out, chain[-1], model)
        start = chain[-1]["control"]
        print(f"resuming at {start['phase']} epoch {start['epoch']} step {start['phase_step']}", flush=True)
    else:
        (out / "recipe.json").write_text(json.dumps(RECIPE, sort_keys=True, indent=1))
    stamping = []

    def boundary(kind, ctl, o):
        entry = {"index": len(chain), "kind": kind, "control": ctl, "state": state_digest(model, o),
                 "model": model_digest(model.state_dict()),
                 "previous": hashlib.sha256(json.dumps(chain[-1], sort_keys=True).encode()).hexdigest() if chain else None,
                 "time": time.time()}
        if kind == "initial":
            entry["environment"] = environment()
            entry["recipe"] = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()
            entry["data"] = {p: hashlib.sha256((Path(a.data) / p / "stream.json").read_bytes()).hexdigest()
                             for p in ("wikipedia", "conversation")}
        # Every boundary checkpoint is kept: each one is the start of a verifiable segment.
        # Checkpoint first, then the chain entry, so a crash between them is harmless.
        tmp = out / "ckpt.tmp"
        torch.save({"control": ctl, "model": model.state_dict(), "opt": o.state_dict()}, tmp)
        os.replace(tmp, out / f"ckpt-{entry['index']:05d}.pt")
        if kind in ("base", "final"):
            torch.save(model.state_dict(), out / f"{'base' if kind == 'base' else 'conversation'}-model.pt")
        chain.append(entry)
        with (out / "chain.jsonl").open("a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"boundary {entry['index']} {kind} step {ctl['global_step']} state {entry['state'][:16]}", flush=True)
        if TIMESTAMP:
            stamping.append(stamp.stamp_async(out, entry))

    run_phases(a.data, model, opt, boundary, start)
    for t in stamping:
        t.join(timeout=120)
    print("training complete", flush=True)


def cmd_fork(a):
    """Fork a published run at boundary K without some Wikipedia documents it had not reached.

    Entries 0..K are copied unchanged, so the fork provably shares that history. Entry K+1
    (kind "fork") keeps the same state and records the removed documents' target ranges in
    the control; `train --out FORK` then continues, skipping exactly those documents."""
    src, out, k = Path(a.run), Path(a.out), a.at
    chain = read_chain(src)
    start = chain[k]["control"]
    if start["phase"] != "wikipedia" or "exclude" in start:
        sys.exit("fork from an unforked Wikipedia boundary")
    pages, ranges, docs = set(a.exclude_page), [], []
    with (Path(a.data) / "wikipedia" / "documents.jsonl").open("rb") as f:
        for line in f:
            d = json.loads(line)
            if str(d["identity"][1]) in pages:
                if d["target_start"] < start["cursor"]:
                    sys.exit(f"page {d['identity'][1]} was already trained before boundary {k}")
                ranges.append([d["target_start"], d["targets"]])
                docs.append(d["identity"])
    if len(docs) != len(pages):
        sys.exit("not every requested page is in the Wikipedia stream")
    out.mkdir(parents=True, exist_ok=False)
    (out / "recipe.json").write_bytes((src / "recipe.json").read_bytes())
    lines = (src / "chain.jsonl").read_text().splitlines()[:k + 1]
    ck = torch.load(src / f"ckpt-{k:05d}.pt", map_location="cpu", weights_only=True)
    torch.save(ck, out / f"ckpt-{k:05d}.pt")
    ctl = {**start, "exclude": {"wikipedia": sorted(ranges)}}
    torch.save({**ck, "control": ctl}, out / f"ckpt-{k + 1:05d}.pt")
    entry = {"index": k + 1, "kind": "fork", "control": ctl, "state": chain[k]["state"], "model": chain[k]["model"],
             "previous": hashlib.sha256(json.dumps(chain[k], sort_keys=True).encode()).hexdigest(),
             "time": time.time(), "fork": {"from_entry": k, "documents": sorted(docs)}}
    (out / "chain.jsonl").write_text("\n".join(lines + [json.dumps(entry, sort_keys=True)]) + "\n")
    read_chain(out)
    if TIMESTAMP:
        stamp.stamp_entry(out, entry)
    print(f"fork at boundary {k}: {len(docs)} document(s), {sum(n for _, n in ranges)} targets removed", flush=True)


def verify_segment(data, run, chain, k):
    """Recompute boundary k -> k+1 from checkpoint k; True if it matches bit for bit."""
    want = chain[k + 1]
    if want["kind"] == "fork":  # same state; the control only adds the recorded exclusion
        model, _ = fresh()
        load_checkpoint(run, want, model)
        added = {key: v for key, v in want["control"].items() if key != "exclude"}
        ok = (want["state"] == chain[k]["state"] and want["model"] == chain[k]["model"] and
              added == chain[k]["control"] and "exclude" not in chain[k]["control"] and
              all(s >= chain[k]["control"]["cursor"] for s, _ in excluded(want["control"])))
        print(f"segment {k}->{k + 1} (fork): {'MATCH' if ok else 'MISMATCH'}", flush=True)
        return ok
    model, _ = fresh()
    opt = load_checkpoint(run, chain[k], model)
    seen = {}

    def compare(kind, ctl, o):
        want = chain[k + 1]
        seen["ok"] = (want["kind"] == kind and want["control"] == ctl and want["state"] == state_digest(model, o))
        return STOP
    t0 = time.time()
    run_phases(data, model, opt, compare, chain[k]["control"])
    ok = seen.get("ok", False)
    print(f"segment {k}->{k + 1} ({chain[k + 1]['kind']}): {'MATCH' if ok else 'MISMATCH'} "
          f"in {time.time() - t0:.0f} s", flush=True)
    return ok


def weights_verdict(chain, hit, ok):
    """Console message for a weights check: a match on a non-released entry is not 'not in the chain'."""
    if ok:
        return f"MATCH chain entry {hit[-1]}"
    if hit:
        return f"chain entry {hit[-1]} ({chain[hit[-1]]['kind']}), not a released model (expected base or final)"
    return "NOT in the chain"


def cmd_verify(a):
    run = Path(a.run)
    chain = read_chain(run)
    report = {"time": time.time(), "environment": environment(), "chain_head": chain[-1]["state"], "checks": []}
    if a.data and chain[0]["data"] != {p: hashlib.sha256((Path(a.data) / p / "stream.json").read_bytes()).hexdigest()
                                       for p in ("wikipedia", "conversation")}:
        sys.exit("prepared data differs from the data the chain trained on")
    env = chain[0].get("environment", {})
    if a.data and env and env != report["environment"]:
        print(f"WARNING: environment differs from training: {env} vs {report['environment']}", flush=True)
    if a.weights:
        sd = torch.load(a.weights, map_location="cpu", weights_only=True)
        hit = [e["index"] for e in chain if e["model"] == model_digest(sd)]
        ok = bool(hit) and chain[hit[-1]]["kind"] in ("base", "final")
        print(f"weights {a.weights}: {weights_verdict(chain, hit, ok)}", flush=True)
        report["checks"].append({"weights": str(a.weights), "chain_entry": hit[-1] if hit else None, "match": ok})
    if a.data:
        model, opt = fresh()  # regenerate the seeded initialization
        ok = state_digest(model, opt) == chain[0]["state"]
        print(f"initialization: {'MATCH' if ok else 'MISMATCH'}", flush=True)
        report["checks"].append({"initialization": ok})
        segments = range(len(chain) - 1) if a.all else list(a.segment or [])
        if a.random:
            pool = [k for k in range(len(chain) - 1) if k not in segments]
            segments = list(segments) + [pool.pop(secrets.randbelow(len(pool))) for _ in range(min(a.random, len(pool)))]
        for k in sorted(segments):
            report["checks"].append({"segment": [k, k + 1], "match": verify_segment(a.data, run, chain, k)})
    report["result"] = "PASS" if all(c.get("match", c.get("initialization")) for c in report["checks"]) else "FAIL"
    with (run / "verifications.jsonl").open("a") as f:
        f.write(json.dumps(report, sort_keys=True) + "\n")
    print(f"verify {report['result']}: {len(report['checks'])} checks; "
          f"{sum('segment' in c for c in report['checks'])} of {len(chain) - 1} segments", flush=True)
    if report["result"] != "PASS":
        sys.exit(1)


def cmd_bench(a):
    model, opt = fresh()
    ctl, n, t0 = initial_control(), 0, None
    for b, _ in prefetch(a.data, "wikipedia", [0, 0]):
        loss, ctl["cursor"], _ = step(model, opt, b, ctl)
        n += 1
        if n == 50:
            if DEV == "cuda": torch.cuda.synchronize()
            t0 = time.time()
        if t0 and time.time() - t0 >= a.seconds:
            break
    if DEV == "cuda": torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / (n - 50)
    total = RECIPE["schedule"]["wikipedia_updates"]
    print(json.dumps({"ms_per_update": round(ms, 2), "updates_timed": n - 50, "loss": loss.item(),
                      **environment(), "projected_train_hours": round(total * ms / 3.6e6, 2),
                      "segments": math.ceil(total / RECIPE["boundary_every"]),
                      "minutes_per_segment": round(RECIPE["boundary_every"] * ms / 6e4, 1)}))


@torch.no_grad()
def cmd_chat(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    model = TinyGPT(**RECIPE["model"]).to(DEV)
    model.load_state_dict(torch.load(Path(a.run) / f"{a.model}-model.pt", map_location=DEV, weights_only=True))
    model.lm_head.weight = model.transformer["wte"].weight
    model.eval()
    ids = [i + OFFSET for i in tok.encode(a.prompt, add_special_tokens=False).ids]
    ids = ([USER] + ids + [EOS, ASSISTANT]) if a.model == "conversation" else ([BOS] + ids)
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    out = []
    for _ in range(a.tokens):
        x = torch.tensor([ids[-RECIPE["context"]:]], device=DEV)
        logits = model(x)[0, -1].float() / a.temperature
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
        if nxt == EOS:
            break
        ids.append(nxt)
        out.append(nxt)
    print(tok.decode([i - OFFSET for i in out if i >= OFFSET]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    t = s.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    v = s.add_parser("verify"); v.add_argument("--run", required=True); v.add_argument("--data")
    v.add_argument("--segment", type=int, nargs="*"); v.add_argument("--random", type=int, default=0)
    v.add_argument("--all", action="store_true"); v.add_argument("--weights")
    fk = s.add_parser("fork"); fk.add_argument("--run", required=True); fk.add_argument("--data", required=True)
    fk.add_argument("--at", type=int, required=True); fk.add_argument("--exclude-page", nargs="+", required=True)
    fk.add_argument("--out", required=True)
    b = s.add_parser("bench"); b.add_argument("--data", required=True); b.add_argument("--seconds", type=int, default=600)
    c = s.add_parser("chat"); c.add_argument("--run", required=True); c.add_argument("--tokenizer", required=True)
    c.add_argument("prompt"); c.add_argument("--model", default="conversation", choices=["base", "conversation"])
    c.add_argument("--tokens", type=int, default=200); c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"train": cmd_train, "fork": cmd_fork, "verify": cmd_verify, "bench": cmd_bench, "chat": cmd_chat}[a.cmd](a)


if __name__ == "__main__":
    main()
