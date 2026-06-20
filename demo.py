#!/usr/bin/env python3
"""OpenVerifiableLLM -- one-command demo.

Runs the whole arc and prints a verdict table:

  1. The claim everyone assumes:  same code + same seed -> same model.
  2. The matrix that tests it:     models x conditions, with a PASS/FAIL spread.
  3. The debate hook:              a run whose LOSS matches fp32 to 1e-6 but whose
                                   BITS do not. Is the right bar identity or tolerance?
  4. The security reframe:         verify-before-load (ed25519) vs the old
                                   torch.load(weights_only=False) code-exec hole.
  5. The sealed artifact:          a non-degenerate Merkle tree over real weights.

  python demo.py                 # auto device; CPU uses a fast smoke config
  python demo.py --full          # full step counts (use on the GPU pod)
  python demo.py --device cuda --cross-gpu-results results/results_A40.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from device import get_device, device_name  # noqa: E402
from experiment import run_one  # noqa: E402
from sweep import DEFAULT_CONDITIONS, annotate_reference, cond_label, print_grid  # noqa: E402
import signing  # noqa: E402

BANNER = "=" * 100
DEMO_MODELS = ["mlp", "gpt10m", "lstm"]


def section(title):
    print("\n\n" + BANNER)
    print(f"  {title}")
    print(BANNER)


def run_curated_matrix(device, overrides, track):
    records = []
    for model in DEMO_MODELS:
        for prec, det in DEFAULT_CONDITIONS:
            rec = run_one(model, "shakespeare", prec, det, device=device,
                          overrides=overrides, track_full=track,
                          keep_artifact=(model == "gpt10m" and prec == "fp32" and det),
                          quiet=False)
            rec["condition"] = cond_label(prec, det)
            records.append(rec)
    annotate_reference(records)
    return records


def print_verdict_table(records, cross_gpu_results=None):
    """The headline table, in the requested shape."""
    cross = {}
    if cross_gpu_results and Path(cross_gpu_results).exists():
        for line in Path(cross_gpu_results).read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            cross[(o["model"], o.get("condition"))] = o.get("param_sha256")

    section("VERDICT TABLE")
    print(f"{'MODEL':<8}{'CONDITION':<14}{'REPRODUCIBLE':<14}{'CROSS-GPU':<12}{'FIRST-DIVERGENCE':<18}")
    print("-" * 66)
    for r in records:
        repro = "-" if r["reproducible"] is None else ("PASS" if r["reproducible"] else "FAIL")
        fd = r["first_divergence_step"]
        first_div = "-" if fd is None else f"step {fd}"
        xg = "-"
        key = (r["model"], r.get("condition"))
        if key in cross:
            xg = "SAME" if cross[key] == r["param_sha256"] else "DIFF"
        print(f"{r['model']:<8}{r['condition']:<14}{repro:<14}{xg:<12}{first_div:<18}")
    print("-" * 66)
    if not cross:
        print("CROSS-GPU column is '-' (single device this run). Fill it by running the same")
        print("sweep on a second GPU type and passing --cross-gpu-results -- see RUNBOOK.md.")


def show_debate_hook(records):
    section("THE DEBATE HOOK  -- is the right bar bitwise identity or numerical tolerance?")
    print("verify() accepts losses within rel_tol=1e-6 but compares parameters by EXACT hash.")
    print("So a run can PASS the loss check yet FAIL the bitwise check:\n")
    hooks = [r for r in records if r["vs_fp32_bitwise"] is False and r["vs_fp32_losstol"] is True]
    diffs = [r for r in records if r["vs_fp32_bitwise"] is False]
    shown = hooks or diffs
    if not shown:
        print("  (No cell diverged from fp32 on this device. On CPU, TF32 is a no-op and bf16")
        print("   may round identically at this scale -- the TF32 divergence is an Ampere-GPU")
        print("   phenomenon. Run on the pod with --device cuda to populate this. See RUNBOOK.md.)")
        return
    for r in shown[:4]:
        tol = "within 1e-6" if r["vs_fp32_losstol"] else "outside 1e-6"
        print(f"  {r['model']:<7} {r['condition']:<12} loss={r['final_loss']:.8f} ({tol} of fp32)"
              f"   bits vs fp32: DIFFER   hash={r['param_sha256'][:16]}")
    if hooks:
        print("\n  ^ These cells are the exhibit: the loss test says 'reproduced', the hash says 'no'.")
    print("\n  Research framing: bitwise identity is a strong, brittle bar; loss-tolerance is a")
    print("  weak, forgiving one. Verifiable training has to pick a bar and defend it.")


def show_security():
    section("SECURITY  -- verify the signature BEFORE you deserialize")
    print("Original load path:  torch.load(path, weights_only=False)  then compare a SHA-256.")
    print("Problem: weights_only=False unpickles -> arbitrary code runs BEFORE the hash check.")
    print("Fix: ed25519-sign the artifact bytes; verify the signature first; a tampered file is")
    print("rejected without ever being unpickled.\n")
    import tempfile
    sk, vk = signing.generate_keypair()
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "checkpoint.bin"
        art.write_bytes(b"\x00trusted-model-weights" * 500)
        signing.sign_file(art, sk)
        print(f"  signed artifact ............ {art.name}  (+ {art.name}.sig)")
        print(f"  clean verify ............... {signing.verify_file(art, verify_key=vk)}")
        art.write_bytes(b"\x00MALICIOUS-PAYLOAD" * 500)  # attacker tampers the file
        try:
            signing.verify_file(art, verify_key=vk)
            print("  tampered verify ............ ERROR: tamper NOT detected")
        except signing.SignatureError:
            print("  tampered verify ............ REJECTED before deserialization  [attack stopped]")


def show_merkle(records):
    section("SEALED ARTIFACT  -- a non-degenerate Merkle tree over real weights")
    gpt = next((r for r in records if r["model"] == "gpt10m"), None)
    if gpt:
        mb = gpt["artifact_size_bytes"] / 1e6
        print(f"  gpt10m checkpoint: {gpt['num_params']:,} params -> {mb:.1f} MB safetensors")
        print(f"  Merkle leaves (1 MB chunks): {gpt['merkle_chunk_count']}   root: {gpt['merkle_root'][:24]}...")
        print(f"  (The original 16-char toy was 21 KB -> 1 chunk: the tree was a single hash.")
        print(f"   At {gpt['merkle_chunk_count']} chunks you can prove ANY chunk against the root.)")


def closing_pitch():
    section("CLOSING  -- for a research group")
    print("""\
  Training reproducibility is assumed everywhere and verified almost nowhere. The
  same seed and the same code do NOT guarantee the same model: TF32 (the Ampere
  default) silently disagrees with fp32, nondeterministic kernels diverge run to
  run, and different GPUs reduce floating point in different orders.

  This tool makes that assumption testable: a parametrized runner, a results
  matrix that surfaces exactly which conditions hold and which break, a divergence
  curve that times the first bit-difference, and a signed-and-Merkle-sealed
  checkpoint so provenance is verifiable rather than asserted.

  The open problem we'd put to the room: what is the right verification bar?
  Bitwise identity is strong but brittle and hardware-bound. Loss-tolerance is
  portable but admits silent precision drift. Pick one and you've defined what
  'reproducible training' even means -- and that choice is still unsettled.""")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--full", action="store_true", help="full step counts (use on GPU)")
    p.add_argument("--track-divergence", action="store_true")
    p.add_argument("--cross-gpu-results", default=None,
                   help="a results.jsonl from a second GPU to fill the CROSS-GPU column")
    args = p.parse_args()

    dev = get_device() if args.device == "auto" else __import__("torch").device(args.device)
    overrides = {}
    if not args.full and dev.type == "cpu":
        overrides.update(total_steps=8, batch_size=8, block_size=64)

    print(BANNER)
    print("  OpenVerifiableLLM -- verifying what everyone assumes about training reproducibility")
    print(BANNER)
    print(f"  device: {dev.type} ({device_name(dev)})")
    if overrides:
        print(f"  (CPU smoke config: {overrides}; use --full on the GPU pod for the real run)")

    records = run_curated_matrix(args.device, overrides, args.track_divergence)
    print_grid(records)
    print_verdict_table(records, args.cross_gpu_results)
    show_debate_hook(records)
    show_security()
    show_merkle(records)
    closing_pitch()


if __name__ == "__main__":
    main()
