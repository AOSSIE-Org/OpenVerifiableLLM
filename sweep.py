#!/usr/bin/env python3
"""Run the models x conditions matrix and print a verdict grid.

Loops the parametrized runner over a deliberate spread of conditions so the grid
contains BOTH reproducible and broken cells -- an all-PASS grid has nothing to
debate. Writes one JSON record per cell to results/results.jsonl.

  python sweep.py                       # default 3 models x 4 conditions, shakespeare
  python sweep.py --quick               # tiny CPU smoke (8 steps) to prove plumbing
  python sweep.py --stretch             # add gpt50m + cnn rows
  python sweep.py --device cuda --track-divergence --datasets shakespeare wikitext

Two distinct verdicts per cell (kept separate on purpose):
  * REPRO       run-to-run on the same hardware (twin runs identical?)
  * vs-FP32     agreement with the per-model fp32+deterministic reference, both
                bitwise (exact hash) and within a 1e-6 loss tolerance. The cell
                where loss agrees but the hash does not is the planted debate hook.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiment import REFERENCE_LOSS_RTOL, append_record, run_one  # noqa: E402

DEFAULT_MODELS = ["mlp", "gpt10m", "lstm"]
# (precision, deterministic) -> deliberately spans PASS and FAIL outcomes.
DEFAULT_CONDITIONS = [("fp32", True), ("fp32", False), ("tf32", True), ("bf16", True)]
COND_LABEL = {
    ("fp32", True): "fp32 det-on",
    ("fp32", False): "fp32 det-off",
    ("tf32", True): "tf32",
    ("bf16", True): "bf16",
    ("fp16", True): "fp16",
}


def cond_label(prec, det):
    return COND_LABEL.get((prec, det), f"{prec} det-{'on' if det else 'off'}")


def annotate_reference(records):
    """Tag each cell with agreement vs the per-(model,dataset) fp32+det reference."""
    refs = {}
    for r in records:
        if r["precision"] == "fp32" and r["deterministic"]:
            refs[(r["model"], r["dataset"])] = r
    for r in records:
        ref = refs.get((r["model"], r["dataset"]))
        r["is_reference"] = ref is not None and r is ref
        if not ref:
            r["vs_fp32_bitwise"] = None
            r["vs_fp32_losstol"] = None
            continue
        r["vs_fp32_bitwise"] = r["param_sha256"] == ref["param_sha256"]
        r["vs_fp32_losstol"] = math.isclose(
            r["final_loss"], ref["final_loss"], rel_tol=REFERENCE_LOSS_RTOL
        )
    return records


def _fmt_repro(r):
    if r["reproducible"] is None:
        return "  -  "
    return "PASS " if r["reproducible"] else "FAIL "


def _fmt_vs(bitwise, losstol):
    if bitwise is None:
        return "ref "
    if bitwise:
        return "SAME"
    return "DIFF"


def print_grid(records):
    print("\n" + "=" * 100)
    print("RESULTS MATRIX  (REPRO = run-to-run on this device;  vs-FP32 = agreement with the")
    print("                fp32+deterministic reference for that model)")
    print("=" * 100)
    header = (f"{'MODEL':<8}{'CONDITION':<14}{'REPRO':<7}{'FIRST-DIV':<11}"
              f"{'vs-FP32 bits':<14}{'loss@1e-6':<11}{'FINAL LOSS':<14}{'MERKLE':<10}")
    groups = {}
    for r in records:
        groups.setdefault((r["model"], r["dataset"]), []).append(r)

    debate_cells = []
    for (model, ds), cells in groups.items():
        print(f"\n-- {model} on {ds} --")
        print(header)
        print("-" * len(header))
        for r in cells:
            bitwise, losstol = r["vs_fp32_bitwise"], r["vs_fp32_losstol"]
            fd = r["first_divergence_step"]
            loss_tol_str = "ref " if losstol is None else ("SAME" if losstol else "DIFF")
            star = ""
            if bitwise is False and losstol is True:
                star = "  <-- loss agrees, bits differ (DEBATE)"
                debate_cells.append(r)
            print(f"{model:<8}{r['condition']:<14}{_fmt_repro(r):<7}"
                  f"{('-' if fd is None else str(fd)):<11}"
                  f"{_fmt_vs(bitwise, losstol):<14}{loss_tol_str:<11}"
                  f"{r['final_loss']:<14.6f}"
                  f"{str(r['merkle_chunk_count']) + ' chunks':<10}{star}")

    print("\n" + "=" * 100)
    n_pass = sum(1 for r in records if r["reproducible"] is True)
    n_fail = sum(1 for r in records if r["reproducible"] is False)
    n_diff = sum(1 for r in records if r["vs_fp32_bitwise"] is False)
    print(f"Cells: {len(records)} | run-to-run reproducible: {n_pass} | "
          f"run-to-run broken: {n_fail} | bit-differ from fp32 reference: {n_diff}")
    if debate_cells:
        print(f"Debate-hook cells (loss within 1e-6 of fp32 but NOT bitwise equal): "
              f"{len(debate_cells)} -> "
              + ", ".join(f"{r['model']}/{r['condition']}" for r in debate_cells))
    print("=" * 100)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="+", default=["shakespeare"])
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="tiny CPU smoke preset: 8 steps, batch 8, block 64")
    p.add_argument("--stretch", action="store_true", help="add gpt50m and cnn rows")
    p.add_argument("--include-fp16", action="store_true")
    p.add_argument("--track-divergence", action="store_true")
    p.add_argument("--out", default="results/results.jsonl")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    overrides = {}
    if args.quick:
        overrides.update(total_steps=8, batch_size=8, block_size=64)
    if args.steps is not None:
        overrides["total_steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.block_size is not None:
        overrides["block_size"] = args.block_size

    models = list(args.models)
    datasets = list(args.datasets)
    if args.stretch:
        for m in ("gpt50m",):
            if m not in models:
                models.append(m)
        if "cnn" not in models:
            models.append("cnn")
            if "cifar" not in datasets:
                datasets.append("cifar")

    conditions = list(DEFAULT_CONDITIONS)
    if args.include_fp16:
        conditions.append(("fp16", True))

    # cnn is a vision model: only run it on cifar, and text models only on text.
    def applicable(model, ds):
        is_vision = model == "cnn"
        return is_vision == (ds == "cifar")

    out_path = args.out
    if out_path:
        # fresh file per sweep
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("")

    records = []
    total_pairs = [(m, d) for d in datasets for m in models if applicable(m, d)]
    i = 0
    for (model, ds) in total_pairs:
        for prec, det in conditions:
            i += 1
            if not args.quiet:
                print(f"\n[{i}] {model} | {ds} | {cond_label(prec, det)}")
            rec = run_one(model, ds, prec, det, device=args.device, overrides=overrides,
                          track_full=args.track_divergence,
                          keep_artifact=(prec == "fp32" and det and model == "gpt10m"),
                          quiet=args.quiet)
            rec["condition"] = cond_label(prec, det)
            records.append(rec)
            if out_path:
                append_record(rec, out_path)

    annotate_reference(records)
    print_grid(records)
    if out_path:
        # rewrite with reference annotations included
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\n ~> {len(records)} records written to {out_path}")


if __name__ == "__main__":
    main()
