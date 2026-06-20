#!/usr/bin/env python3
"""Train ONE matrix cell and emit a JSON record.

The single parametrized runner the whole project is built around. Examples:

  # control cell (should be run-to-run reproducible)
  python run_experiment.py --model gpt10m --dataset shakespeare \
      --precision fp32 --deterministic on --out results/results.jsonl

  # the silent-killer cell (Ampere default; diverges from fp32 reference)
  python run_experiment.py --model gpt10m --precision tf32 --deterministic on

  # determinism OFF (run-to-run divergence on GPU; verify on the pod)
  python run_experiment.py --model gpt10m --precision fp32 --deterministic off \
      --track-divergence

Record schema:
  {model, dataset, precision, deterministic, device, final_loss, param_sha256,
   merkle_root, first_divergence_step, reproducible, num_params, ...}
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiment import append_record, run_one  # noqa: E402

MODELS = ["mlp", "gpt10m", "gpt50m", "lstm", "cnn"]
DATASETS = ["shakespeare", "wikitext", "enwik8", "cifar"]
PRECISIONS = ["fp32", "tf32", "bf16", "fp16"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--dataset", default="shakespeare", choices=DATASETS)
    p.add_argument("--precision", default="fp32", choices=PRECISIONS)
    p.add_argument("--deterministic", default="on", choices=["on", "off"])
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--steps", type=int, default=None, help="override total_steps")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--track-divergence", action="store_true",
                   help="hash params every step in both twin runs (T7 data)")
    p.add_argument("--keep-artifact", action="store_true",
                   help="keep the .safetensors instead of deleting after hashing")
    p.add_argument("--no-twin", action="store_true",
                   help="skip the run-to-run twin run (faster; no reproducibility verdict)")
    p.add_argument("--out", default=None, help="append the JSON record to this file")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Validate model/dataset compatibility
    if args.model == "cnn" and args.dataset != "cifar":
        sys.exit(f"Error: model '{args.model}' is only compatible with dataset 'cifar', not '{args.dataset}'")
    if args.model != "cnn" and args.dataset == "cifar":
        sys.exit(f"Error: dataset 'cifar' is only compatible with model 'cnn', not '{args.model}'")

    overrides = {}
    if args.steps is not None:
        overrides["total_steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.block_size is not None:
        overrides["block_size"] = args.block_size

    record = run_one(
        args.model, args.dataset, args.precision, args.deterministic, args.seed,
        device=args.device, overrides=overrides, track_full=args.track_divergence,
        keep_artifact=args.keep_artifact, twin=not args.no_twin, quiet=args.quiet,
    )
    print(json.dumps(record, indent=2))
    if args.out:
        append_record(record, args.out)
        print(f"\n ~> record appended to {args.out}")


if __name__ == "__main__":
    main()
