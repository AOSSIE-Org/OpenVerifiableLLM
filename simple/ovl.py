"""OpenVerifiableLLM, simple production loop: train -> replay -> chat.

One file, one GPU. Reuses the project's prepared streams, batch order, model and
initialization, so the guarantees that matter stay intact:
  G03 reproducible init     ovl_pipeline.training.initialize (seeded, CPU, no pretrained weights)
  G04 exactly one pass      every target id checked in order; final cursor == stream targets
  G05 committed progress    chain.jsonl: hash-linked state digests at fixed boundaries
  G06 conversation phase    same loop on the conversation stream, fresh AdamW at transition
  G07 exact replay          `replay` recomputes from init and must match every boundary bit for bit

Usage (on the pod):
  python simple/ovl.py train  --data PREP --out RUN
  python simple/ovl.py replay --data PREP --run RUN
  python simple/ovl.py chat   --run RUN --tokenizer PREP/tokenizer/tokenizer.json "What is Wikipedia?"
  python simple/ovl.py bench  --data PREP --seconds 600
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch
import torch.nn.functional as F

from ovl_pipeline.data import BOS, EOS, USER, ASSISTANT, OFFSET, batches
from ovl_pipeline import training
from model import TinyGPT

RECIPE = {"batch_size": 32, "boundary_every": 18080, "context": 512, "init": "normal-0.02-tied-v1",
          "learning_rate": "0.0003", "seed": 1234, "weight_decay": "0.01",
          "model": {"attn_impl": "manual", "dropout": 0, "embed_dim": 384, "max_seq_len": 512,
                    "num_heads": 6, "num_layers": 6, "vocab_size": 32000}}
PHASES = ("wikipedia", "conversation")
DEV = os.environ.get("OVL_DEVICE", "cuda")  # "cpu" only for smoke tests
if os.environ.get("OVL_BOUNDARY_EVERY"):
    RECIPE["boundary_every"] = int(os.environ["OVL_BOUNDARY_EVERY"])


def deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=float(RECIPE["learning_rate"]),
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
    torch.set_num_threads(os.cpu_count())
    model.to(DEV)
    model.lm_head.weight = model.transformer["wte"].weight
    return model, optimizer(model)


def state_digest(model, opt):
    h = hashlib.sha256()
    tensors = [(f"model.{k}", v) for k, v in model.state_dict().items()]
    for i, p in enumerate(model.parameters()):
        for k, v in sorted(opt.state.get(p, {}).items()):
            tensors.append((f"opt.{i}.{k}", v if torch.is_tensor(v) else torch.tensor(v)))
    for name, t in tensors:
        t = t.detach().reshape(-1).contiguous().cpu()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def prefetch(data, phase, skip=0):
    """Same batch order as ovl_pipeline.data.batches, produced on a background thread."""
    q = queue.Queue(maxsize=64)

    def work():
        try:
            for i, b in enumerate(batches(Path(data) / phase, RECIPE["context"], RECIPE["batch_size"])):
                if i >= skip:
                    q.put(b)
        except BaseException as e:  # surface loader errors in the main thread
            q.put(e)
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (b := q.get()) is not None:
        if isinstance(b, BaseException):
            raise b
        yield b


def step(model, opt, b, cursor):
    ids = b["target_ids"][b["mask"]]
    if ids.numel() == 0 or ids[0].item() != cursor or ids[-1].item() != cursor + ids.numel() - 1:
        raise SystemExit(f"coverage error at cursor {cursor}")  # gap, repeat or reorder
    x, y, m = (b[k].to(DEV, non_blocking=True) for k in ("inputs", "targets", "mask"))
    with torch.autocast(DEV, dtype=torch.bfloat16, cache_enabled=False):
        logits = model(x)
        losses = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction="none")
        m = m.flatten()
        loss = losses[m].sum() / m.sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss, cursor + ids.numel(), hashlib.sha256(b["inputs"].numpy().tobytes()).digest()


def targets(data, phase):
    return json.loads((Path(data) / phase / "stream.json").read_text())["targets"]


def run_phases(data, model, opt, on_boundary, start=None):
    """Drive both phases; call on_boundary(kind, control) at every scheduled boundary."""
    ctl = start or {"phase": "wikipedia", "global_step": 0, "phase_step": 0, "cursor": 0,
                    "batches": hashlib.sha256(b"ovl.simple.batches.v1").hexdigest()}
    if start is None:
        on_boundary("initial", ctl)
    for phase in PHASES:
        if PHASES.index(phase) < PHASES.index(ctl["phase"]):
            continue
        if phase == "conversation" and ctl["phase"] != phase:
            opt = optimizer(model)
            ctl = {**ctl, "phase": phase, "phase_step": 0, "cursor": 0}
            on_boundary("transition", ctl, opt)
        total, t0, loss = targets(data, phase), time.time(), None
        for b in prefetch(data, phase, skip=ctl["phase_step"]):
            loss, cursor, bh = step(model, opt, b, ctl["cursor"])
            ctl = {**ctl, "global_step": ctl["global_step"] + 1, "phase_step": ctl["phase_step"] + 1,
                   "cursor": cursor, "batches": hashlib.sha256(bytes.fromhex(ctl["batches"]) + bh).hexdigest()}
            if ctl["phase_step"] % 1000 == 0:
                lv = loss.item()
                if lv != lv or lv in (float("inf"), float("-inf")):
                    raise SystemExit(f"nonfinite loss at step {ctl['global_step']}")
                rate = ctl["phase_step"] / (time.time() - t0)
                print(f"{phase} step {ctl['phase_step']} loss {lv:.4f} {1000 / rate:.1f} ms/update", flush=True)
            end = ctl["cursor"] == total
            if end or ctl["phase_step"] % RECIPE["boundary_every"] == 0:
                kind = ("final" if phase == "conversation" else "base") if end else "progress"
                on_boundary(kind, ctl, opt)
        if ctl["cursor"] != total:
            raise SystemExit(f"{phase}: covered {ctl['cursor']} of {total} targets")
    return ctl


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chain_path = out / "chain.jsonl"
    chain = [json.loads(l) for l in chain_path.read_text().splitlines()] if chain_path.exists() else []
    model, opt = fresh()
    start = None
    if chain:
        last = chain[-1]
        if last["kind"] == "final":
            sys.exit("run already complete")
        ck = torch.load(out / f"ckpt-{last['index']:05d}.pt", map_location=DEV, weights_only=False)
        if ck["control"] != last["control"]:
            sys.exit("checkpoint does not match the last chain entry")
        model.load_state_dict(ck["model"])
        model.lm_head.weight = model.transformer["wte"].weight
        if last["control"]["phase"] == "conversation":
            opt = optimizer(model)
        opt.load_state_dict(ck["opt"])
        if state_digest(model, opt) != last["state"]:
            sys.exit("checkpoint state digest differs from the chain")
        start = last["control"]
        print(f"resuming at {start['phase']} step {start['phase_step']}", flush=True)
    current = {"opt": opt}

    def boundary(kind, ctl, new_opt=None):
        if new_opt is not None:
            current["opt"] = new_opt
        o = current["opt"]
        entry = {"index": len(chain), "kind": kind, "control": ctl, "state": state_digest(model, o),
                 "previous": hashlib.sha256(json.dumps(chain[-1], sort_keys=True).encode()).hexdigest() if chain else None,
                 "time": time.time()}
        # Checkpoint first, then the chain entry; a crash between them leaves the
        # previous checkpoint usable. Only the last two are kept (the chain keeps every digest).
        tmp = out / "ckpt.tmp"
        torch.save({"control": ctl, "model": model.state_dict(), "opt": o.state_dict()}, tmp)
        os.replace(tmp, out / f"ckpt-{entry['index']:05d}.pt")
        if kind in ("base", "final"):
            torch.save(model.state_dict(), out / f"{ctl['phase'] if kind == 'final' else 'base'}-model.pt")
        chain.append(entry)
        with chain_path.open("a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        old = out / f"ckpt-{entry['index'] - 2:05d}.pt"
        if old.exists():
            old.unlink()
        print(f"boundary {entry['index']} {kind} step {ctl['global_step']} state {entry['state'][:16]}", flush=True)

    run_phases(a.data, model, opt, boundary, start)
    (out / "recipe.json").write_text(json.dumps(RECIPE, sort_keys=True, indent=1))
    print("training complete", flush=True)


def cmd_replay(a):
    run = Path(a.run)
    chain = [json.loads(l) for l in (run / "chain.jsonl").read_text().splitlines()]
    if chain[-1]["kind"] != "final":
        sys.exit("replay needs a completed run")
    model, opt = fresh()
    pos, results, current = 0, [], {"opt": opt}

    def compare(kind, ctl, new_opt=None):
        nonlocal pos
        if new_opt is not None:
            current["opt"] = new_opt
        want = chain[pos]
        got = state_digest(model, current["opt"])
        ok = want["kind"] == kind and want["control"] == ctl and want["state"] == got
        results.append({"index": pos, "kind": kind, "step": ctl["global_step"], "state": got, "match": ok})
        print(f"replay boundary {pos} {kind} {'MATCH' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            (run / "replay.json").write_text(json.dumps({"result": "FAIL", "boundaries": results}, indent=1))
            sys.exit(f"replay mismatch at boundary {pos}")
        pos += 1

    run_phases(a.data, model, opt, compare)
    ok = pos == len(chain)
    (run / "replay.json").write_text(json.dumps({"result": "PASS" if ok else "FAIL", "boundaries": results}, indent=1))
    print("replay", "PASS" if ok else "FAIL: chain has extra boundaries", flush=True)


def cmd_bench(a):
    model, opt = fresh()
    phase, cursor, n, t0 = "wikipedia", 0, 0, None
    for b in prefetch(a.data, phase):
        loss, cursor, _ = step(model, opt, b, cursor)
        n += 1
        if n == 50:
            torch.cuda.synchronize()
            t0 = time.time()
        if t0 and time.time() - t0 >= a.seconds:
            break
    if DEV == "cuda": torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / (n - 50)
    total = 560177 + 147
    print(json.dumps({"ms_per_update": round(ms, 2), "updates_timed": n - 50, "loss": loss.item(),
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "projected_train_hours": round(total * ms / 3.6e6, 2),
                      "projected_train_plus_replay_hours": round(2 * total * ms / 3.6e6, 2)}))


@torch.no_grad()
def cmd_chat(a):
    from tokenizers import Tokenizer
    from model import TinyGPT
    tok = Tokenizer.from_file(a.tokenizer)
    model = TinyGPT(**RECIPE["model"]).to(DEV)
    model.load_state_dict(torch.load(Path(a.run) / f"{a.model}-model.pt", map_location=DEV))
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
    r = s.add_parser("replay"); r.add_argument("--data", required=True); r.add_argument("--run", required=True)
    b = s.add_parser("bench"); b.add_argument("--data", required=True); b.add_argument("--seconds", type=int, default=600)
    c = s.add_parser("chat"); c.add_argument("--run", required=True); c.add_argument("--tokenizer", required=True)
    c.add_argument("prompt"); c.add_argument("--model", default="conversation", choices=["base", "conversation"])
    c.add_argument("--tokens", type=int, default=200); c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"train": cmd_train, "replay": cmd_replay, "bench": cmd_bench, "chat": cmd_chat}[a.cmd](a)


if __name__ == "__main__":
    main()
