"""Decoded-content review of training checkpoints before publication.

For every ckpt-XXXXX.pt in RUN: exact sha256, zip member names, restricted load
(weights_only=True), exact structure (control/model/opt), every string in the decoded
object against an allowlist of field names, and the binding to chain.jsonl (control and
state digest equal the committed entry). Prints one JSON line per file; exit 1 on any
finding.
    OVL_DEVICE=cpu python simple/review_ckpt.py RUN
"""
import hashlib
import json
import pickletools
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ovl  # noqa: E402
import torch  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CONTROL = {"phase": str, "epoch": int, "global_step": int, "phase_step": int, "cursor": int, "pos": list, "batches": str}
ADAM_STATE = {"step", "exp_avg", "exp_avg_sq"}
ADAM_GROUP = {"lr", "betas", "eps", "weight_decay", "amsgrad", "foreach", "maximize", "capturable",
              "differentiable", "fused", "params", "decoupled_weight_decay"}
GLOBALS = {"collections OrderedDict", "torch FloatStorage", "torch._utils _rebuild_tensor_v2"}
MODULES = set()
allowed_fields = set()
MEMBER = re.compile(r"^[A-Za-z0-9_.\-]+/(data\.pkl|data/\d+|version|byteorder|\.data/serialization_id|\.format_version|\.storage_alignment)$")


def strings(obj, out):
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            strings(k, out)
            strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strings(v, out)
    return out


def review(path, entry, model_keys):
    problems = []
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    bad_members = [n for n in names if not MEMBER.match(n)]
    if bad_members:
        problems.append(f"unexpected zip members {bad_members[:5]}")
    prefixes = {n.split("/", 1)[0] for n in names}
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if set(ck) != {"control", "model", "opt"}:
        problems.append(f"top-level keys {sorted(ck)}")
    ctl = ck.get("control", {})
    excl = ctl.get("exclude")  # recorded by a fork: {phase: [[target_start, count], ...]}
    if excl is not None and not (isinstance(excl, dict) and set(excl) <= {"wikipedia", "conversation"} and all(
            isinstance(r, list) and len(r) == 2 and all(isinstance(v, int) for v in r) for rs in excl.values() for r in rs)):
        problems.append("exclude field")
    if set(ctl) - {"exclude"} != set(CONTROL) or any(not isinstance(ctl[k], t) for k, t in CONTROL.items()):
        problems.append("control fields")
    elif ctl["phase"] not in ("wikipedia", "conversation") or not HEX64.match(ctl["batches"]) \
            or len(ctl["pos"]) != 2 or not all(isinstance(v, int) for v in ctl["pos"]):
        problems.append("control values")
    model = ck.get("model", {})
    if sorted(model) != sorted(model_keys) or not all(torch.is_tensor(v) for v in model.values()):
        problems.append("model state names/types")
    opt = ck.get("opt", {})
    if set(opt) != {"state", "param_groups"}:
        problems.append("optimizer keys")
    else:
        for s in opt["state"].values():
            if set(s) != ADAM_STATE or not all(torch.is_tensor(v) for v in s.values()):
                problems.append("optimizer state")
                break
        for g in opt["param_groups"]:
            if not set(g) <= ADAM_GROUP:
                problems.append(f"optimizer group keys {sorted(set(g) - ADAM_GROUP)}")
    allowed = allowed_fields | set(model_keys)
    # Raw pickle: only the three tensor-rebuild globals, and no text beyond field, parameter and
    # module names, device labels, digits and hex digests (torch keeps module names in _metadata).
    with zipfile.ZipFile(path) as z:
        pkl = z.read(next(n for n in names if n.endswith("/data.pkl")))
    raw_ok = set(model_keys) | MODULES | {"_metadata", "version", "storage", "cpu", "cuda:0"}
    for op, arg, _ in pickletools.genops(pkl):
        if op.name == "GLOBAL" and arg not in GLOBALS:
            problems.append(f"pickle global {arg}")
        elif isinstance(arg, str) and op.name != "GLOBAL" and arg not in raw_ok and arg not in allowed_fields \
                and not HEX64.match(arg) and not arg.isdigit():
            problems.append(f"raw pickle string {arg[:40]!r}")
            break
    found = strings(ck, set())
    extra = sorted(s for s in found if s not in allowed and not HEX64.match(s))
    if extra:
        problems.append(f"unexpected strings {extra[:5]}")
    # Binding to the committed chain entry (same checks the verifier makes).
    if ctl != entry["control"]:
        problems.append("control differs from chain")
    else:
        m, _ = ovl.fresh()
        opt_obj = ovl.load_checkpoint(path.parent, entry, m)  # exits on a state-digest mismatch
        del opt_obj
    return {"file": path.name, "sha256": sha, "bytes": len(data), "zip_prefixes": sorted(prefixes),
            "zip_members": len(names), "tensors": len(model) + sum(len(s) for s in opt.get("state", {}).values()),
            "strings_outside_allowlist": extra, "chain_entry": entry["index"], "chain_kind": entry["kind"],
            "global_step": ctl.get("global_step"), "result": "PASS" if not problems else "FAIL", "problems": problems}


def review_model(path, entry, model_keys):
    """base-model.pt / conversation-model.pt: a bare state_dict equal to the committed model."""
    problems = []
    data = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        bad = [n for n in names if not MEMBER.match(n)]
        pkl = z.read(next(n for n in names if n.endswith("/data.pkl")))
    if bad:
        problems.append(f"unexpected zip members {bad[:5]}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if sorted(sd) != sorted(model_keys) or not all(torch.is_tensor(v) for v in sd.values()):
        problems.append("state names/types")
    raw_ok = set(model_keys) | MODULES | {"_metadata", "version", "storage", "cpu", "cuda:0"}
    for op, arg, _ in pickletools.genops(pkl):
        if op.name == "GLOBAL" and arg not in GLOBALS:
            problems.append(f"pickle global {arg}")
        elif isinstance(arg, str) and op.name != "GLOBAL" and arg not in raw_ok and not arg.isdigit():
            problems.append(f"raw pickle string {arg[:40]!r}")
            break
    if ovl.model_digest(sd) != entry["model"]:
        problems.append("weights differ from the committed chain entry")
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "zip_prefixes": sorted({n.split("/", 1)[0] for n in names}), "zip_members": len(names),
            "tensors": len(sd), "chain_entry": entry["index"], "chain_kind": entry["kind"],
            "global_step": entry["control"]["global_step"], "strings_outside_allowlist": [],
            "result": "PASS" if not problems else "FAIL", "problems": problems}


def main():
    run = Path(sys.argv[1])
    chain = ovl.read_chain(run)
    probe = ovl.TinyGPT(**ovl.RECIPE["model"])
    model_keys = list(probe.state_dict())
    MODULES.update(name for name, _ in probe.named_modules())
    allowed_fields.update(set(CONTROL) | ADAM_STATE | ADAM_GROUP | {"control", "model", "opt", "state", "param_groups",
                                                                   "wikipedia", "conversation", "exclude"})
    failed = False
    for entry in chain:
        path = run / f"ckpt-{entry['index']:05d}.pt"
        if not path.exists():
            continue
        r = review(path, entry, model_keys)
        failed |= r["result"] != "PASS"
        print(json.dumps(r, sort_keys=True), flush=True)
    for name, kind in (("base-model.pt", "base"), ("conversation-model.pt", "final")):
        entry = next((e for e in chain if e["kind"] == kind), None)
        if entry and (run / name).exists():
            r = review_model(run / name, entry, model_keys)
            failed |= r["result"] != "PASS"
            print(json.dumps(r, sort_keys=True), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
