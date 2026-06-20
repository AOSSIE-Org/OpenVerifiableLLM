"""Segmented-replay audit (prover vs auditor) on the scaled model + real data.

This is now just one cell of the larger matrix, but it is the cell that proves the
core mechanism: a prover trains 0..N and seals a mid-checkpoint; an auditor resumes
from the checkpoint and re-runs N/2..N; if the replay is bitwise deterministic the
two telemetry trajectories and the final parameter hashes agree.

What changed from the toy:
  * gpt10m (~11M params) on char-level Shakespeare instead of a 16-char string.
  * get_batch is the FIRST statement inside the training loop, so the batch draw
    rides the same global torch RNG the checkpoint already saves/restores -- move
    it out of the loop and the replay silently breaks.
  * Checkpoints are ed25519-signed; the auditor verifies the signature BEFORE
    deserializing (verified_torch_load), closing the torch.load(weights_only=False)
    code-execution hole. The broken-seal scenario now demonstrates that fix.

Fast CPU smoke run (keeps full model dims so the Merkle tree stays non-degenerate,
but shrinks the workload):

    OVL_TOTAL_STEPS=10 OVL_CHECKPOINT_STEP=5 OVL_BATCH_SIZE=4 OVL_BLOCK_SIZE=64 \
        python reproducibility.py
"""

import json
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from config import model_config
from dataset import get_dataset
from device import accel_rng_state, device_name, get_device, restore_accel_rng_state
from main import set_seed
from model import build_model, count_params
from signing import SignatureError, sign_file, signed_torch_save, verified_torch_load
from telemetry import TelemetryLogger
from artifacts import (
    CHECKPOINT_MERKLE_PATH,
    CHECKPOINT_WEIGHTS_PATH,
    MERKLE_CHUNK_SIZE_BYTES,
    model_parameters_sha256,
    save_model_safetensors,
    write_merkle_manifest,
)

# CUDA when available, else CPU. The same code path runs on both; only the
# floating-point reduction order (the hardware entropy under study) differs.
DEVICE = get_device()

# The audit's canonical cell: gpt10m on shakespeare. Workload knobs (steps/batch/
# block) are env-overridable via OVL_* for a fast CPU smoke; architecture is fixed.
MODEL_NAME = "gpt10m"
DATASET_NAME = os.environ.get("OVL_DATASET", "shakespeare")
CFG = model_config(MODEL_NAME)
CP_STEP = CFG["checkpoint_step"]
TOT_STEP = CFG["total_steps"]
BATCH = CFG["batch_size"]
BLOCK = CFG["block_size"]
CHECKPOINT_STATE_FILE = "mid_checkpoint.pt"


def hash_model(model):
    return model_parameters_sha256(model)


def _build():
    """Construct the (dataset, model, optimizer) triple for one segment."""
    dataset = get_dataset(DATASET_NAME, block_size=BLOCK)
    model = build_model(MODEL_NAME, dataset.vocab_size, CFG).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    return dataset, model, optimizer


def run_training_segment(start_step, end_step, checkpoint_path_to_load=None,
                         log_file="audit.jsonl", seed=None):
    active_seed = seed if seed is not None else CFG["seed"]

    if not checkpoint_path_to_load:
        set_seed(active_seed)

    dataset, model, optimizer = _build()
    logger = TelemetryLogger(filepath=log_file)

    if checkpoint_path_to_load:
        # SECURITY: verify the ed25519 signature over the raw bytes BEFORE any
        # deserialization. A tampered file raises here and never reaches the
        # unpickler (the old code ran torch.load(weights_only=False) first).
        checkpoint = verified_torch_load(checkpoint_path_to_load, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])

        # Secondary, defense-in-depth: the embedded weight hash (now redundant
        # with the signature, but cheap and a nice cross-check).
        if "checkpoint_hash" in checkpoint:
            loaded_hash = logger.hash_model(model)
            if loaded_hash != checkpoint["checkpoint_hash"]:
                print("\n  ALERT: embedded weight hash mismatch after load.")
                print(f"    Expected: {checkpoint['checkpoint_hash'][:16]}...")
                print(f"    Got:      {loaded_hash[:16]}...\n")

        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng_state"])
        restore_accel_rng_state(checkpoint.get("accel_rng_state"))  # CUDA/XPU dropout RNG
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        print(f" ~> Auditor verified signature and restored RNG state at step {start_step}")

    for step in range(start_step, end_step):
        # FIRST line of the loop: batch draw consumes the global torch RNG, which
        # the checkpoint captures. This is what keeps the replay bitwise-exact.
        x, y = dataset.get_batch(BATCH, BLOCK, device=DEVICE)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_step(step, loss.item(), model)

        if not checkpoint_path_to_load and step == (CP_STEP - 1):
            current_model_hash = logger.hash_model(model)
            weights_path = save_model_safetensors(
                model,
                CHECKPOINT_WEIGHTS_PATH,
                metadata={
                    "format": "pt-state-dict",
                    "tensor_sha256": current_model_hash,
                    "checkpoint_step": str(CP_STEP),
                },
            )
            sign_file(weights_path)  # sign the stable safetensors artifact too
            merkle_manifest = write_merkle_manifest(
                weights_path, CHECKPOINT_MERKLE_PATH, chunk_size=MERKLE_CHUNK_SIZE_BYTES,
            )

            # signed_torch_save = torch.save + ed25519 sign of the raw bytes.
            signed_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "accel_rng_state": accel_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "checkpoint_hash": current_model_hash,
                    "safetensors_path": str(weights_path),
                    "safetensors_sha256": merkle_manifest["sha256"],
                    "safetensors_merkle_root": merkle_manifest["merkle_root"],
                    "merkle_chunk_size_bytes": merkle_manifest["chunk_size_bytes"],
                },
                CHECKPOINT_STATE_FILE,
            )
            print(f" ~> Prover sealed + SIGNED checkpoint at step {CP_STEP}")
            print(f" ~> Stable weights: {weights_path} | Merkle root: "
                  f"{merkle_manifest['merkle_root'][:16]}... "
                  f"({merkle_manifest['chunk_count']} chunks)")

    return model


def bad_seed_auditor(log_file="bad_seed_log.jsonl"):
    """Test 1: correct checkpoint, WRONG seed -> batches and dropout diverge."""
    dataset, model, optimizer = _build()
    logger = TelemetryLogger(filepath=log_file)

    checkpoint = verified_torch_load(CHECKPOINT_STATE_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    set_seed(42)  # BAD SEED (never 42)
    print(" ~> Tampered auditor loaded checkpoint with BAD seed (42)")

    for step in range(CP_STEP, TOT_STEP):
        x, y = dataset.get_batch(BATCH, BLOCK, device=DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.log_step(step, loss.item(), model)
    return model


def secret_noise_auditor(log_file="secret_noise_log.jsonl"):
    """Test 2: correct replay, but secret noise injected into gradients.

    RNG state is restored exactly (like the clean auditor) so the ONLY difference
    from a clean replay is the injected noise -- isolating its effect.
    """
    dataset, model, optimizer = _build()
    logger = TelemetryLogger(filepath=log_file)

    checkpoint = verified_torch_load(CHECKPOINT_STATE_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_state"])
    restore_accel_rng_state(checkpoint.get("accel_rng_state"))
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    print(" ~> Auditor replayed correctly but will inject secret gradient noise")

    for step in range(CP_STEP, TOT_STEP):
        x, y = dataset.get_batch(BATCH, BLOCK, device=DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 1e-10  # tiny secret noise
        optimizer.step()
        logger.log_step(step, loss.item(), model)
    return model


def sabotage_auditor(log_file="post_sabotage_log.jsonl"):
    """Test 3: correct replay, but weights silently mutated AFTER training ends.

    The logged trajectory matches the prover (clean replay), so the loss check
    PASSES -- but the final parameter hash differs. This is exactly the gap the
    debate hook is about: telemetry agreement does not imply bitwise identity.
    """
    dataset, model, optimizer = _build()
    logger = TelemetryLogger(filepath=log_file)

    checkpoint = verified_torch_load(CHECKPOINT_STATE_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_state"])
    restore_accel_rng_state(checkpoint.get("accel_rng_state"))
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    print(" ~> Post-sabotage auditor replayed correctly")

    for step in range(CP_STEP, TOT_STEP):
        x, y = dataset.get_batch(BATCH, BLOCK, device=DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.log_step(step, loss.item(), model)

    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p) * 1e-6  # silent post-training mutation

    print(" ~> Weights silently mutated after training completed")
    return model


def broken_seal_auditor(log_file="broken_seal_log.jsonl"):
    """Test 4: attacker tampers the checkpoint FILE on disk.

    With ed25519 signing this is caught BEFORE deserialization: the stale
    signature no longer matches the mutated bytes, so verified_torch_load raises
    and the malicious file never reaches the unpickler. Returns (model, secure).
    """
    corrupted = "corrupted_checkpoint.pt"
    shutil.copy(CHECKPOINT_STATE_FILE, corrupted)
    shutil.copy(CHECKPOINT_STATE_FILE + ".sig", corrupted + ".sig")  # stale signature
    # Flip a run of bytes in the middle of the artifact (true file tampering).
    with open(corrupted, "r+b") as f:
        f.seek(2048)
        chunk = f.read(32)
        f.seek(2048)
        f.write(bytes((b ^ 0xFF) for b in chunk))
    print(f" ~> Attacker corrupted checkpoint bytes -> {corrupted}")

    try:
        model = run_training_segment(
            start_step=CP_STEP, end_step=TOT_STEP,
            checkpoint_path_to_load=corrupted, log_file=log_file,
        )
        print(" ~> WARNING: corrupted file loaded WITHOUT detection (signing disabled?)")
        return model, False
    except SignatureError as exc:
        print(f" ~> [SECURITY PASS] tampered checkpoint REJECTED before deserialization:\n     {exc}")
        return None, True


def verify(prover_segment, auditor_logs, prover_hash, auditor_hash, label="AUDIT"):
    """Shared verification logic with drift quantification and a cryptographic anchor.

    NOTE (the planted debate hook): losses are compared with rel_tol=1e-6 but the
    parameter hash is compared EXACTLY. A run can therefore pass the loss check and
    fail the hash check. verify() returns the *loss/telemetry* verdict; the hash
    mismatch is reported but does not flip the return value -- which is precisely
    the ambiguity worth arguing about.
    """
    print(f"\n[Verifying: {label}]")

    if len(prover_segment) != len(auditor_logs):
        print(f"Log length mismatch: prover={len(prover_segment)}, auditor={len(auditor_logs)}")
        return False

    match = True
    for p, a in zip(prover_segment, auditor_logs):
        step_match = p["step"] == a["step"]
        loss_match = math.isclose(p["loss"], a["loss"], rel_tol=1e-6)
        grad_match = math.isclose(p["grad_norm"], a["grad_norm"], rel_tol=1e-6)
        param_match = math.isclose(p["param_norm"], a["param_norm"], rel_tol=1e-6)
        step_ok = step_match and loss_match and grad_match and param_match

        if not step_ok:
            match = False
            delta = abs(p["loss"] - a["loss"])
            print(f"Step {p['step']} | Prover: {p['loss']:.8f} | Auditor: {a['loss']:.8f} "
                  f"| delta {delta:.2e} FAILED")
        else:
            print(f"Step {p['step']} | Prover: {p['loss']:.8f} | Auditor: {a['loss']:.8f} | PASSED")

    hash_match = prover_hash == auditor_hash
    if not hash_match:
        print(f"\n Hash mismatch! Prover: {prover_hash[:16]} // Auditor: {auditor_hash[:16]} "
              f"[HASH ERROR -- loss check may still PASS: that is the debate]")

    if match and hash_match:
        print(f"\n [PASS] {label}: Segment replay is bitwise deterministic.")
    elif match and not hash_match:
        print(f"\n [LOSS-PASS / HASH-FAIL] {label}: trajectory matches but bits differ.")
    else:
        print(f"\n [FAIL] {label}: Trajectories diverged.")

    return match


if __name__ == "__main__":
    print(f"\n=== Device: {DEVICE.type.upper()} ({device_name(DEVICE)}) | torch {torch.__version__} ===")
    print(f"    Model: {MODEL_NAME} (~{count_params(build_model(MODEL_NAME, 128, CFG)):,} params) "
          f"on {DATASET_NAME} | steps {TOT_STEP}, checkpoint @ {CP_STEP}, batch {BATCH}, block {BLOCK}")
    if DEVICE.type == "cuda":
        print("    Strict GPU determinism (cuDNN deterministic + pinned cuBLAS workspace)")

    # Baseline: should PASS
    print("\n Scenario 1: CLEAN AUDIT ")
    prover_model = run_training_segment(0, TOT_STEP, log_file="prover_log.jsonl")
    auditor_model = run_training_segment(CP_STEP, TOT_STEP,
                                         checkpoint_path_to_load=CHECKPOINT_STATE_FILE,
                                         log_file="auditor_log.jsonl")

    with open("prover_log.jsonl") as f:
        prover_logs = [json.loads(line) for line in f]
    with open("auditor_log.jsonl") as f:
        auditor_logs = [json.loads(line) for line in f]

    clean_ok = verify(prover_logs[CP_STEP:TOT_STEP], auditor_logs,
                      hash_model(prover_model), hash_model(auditor_model), label="CLEAN AUDIT")

    # Test 1: Bad seed -> should FAIL
    print("\n Scenario 2: BAD SEED")
    bad_model = bad_seed_auditor()
    with open("bad_seed_log.jsonl") as f:
        tampered_logs = [json.loads(line) for line in f]
    verify(prover_logs[CP_STEP:TOT_STEP], tampered_logs,
           hash_model(prover_model), hash_model(bad_model), label="BAD SEED AUDIT")

    # Test 2: Gradient noise -> should FAIL
    print("\n Scenario 3: NOISE INJECTED")
    noisy_model = secret_noise_auditor()
    with open("secret_noise_log.jsonl") as f:
        noisy_logs = [json.loads(line) for line in f]
    verify(prover_logs[CP_STEP:TOT_STEP], noisy_logs,
           hash_model(prover_model), hash_model(noisy_model), label="NOISY WEIGHTS AUDIT")

    # Test 3: Post-training sabotage -> loss PASS, hash FAIL (the debate)
    print("\n Scenario 4: POST-TRAINING WEIGHT SABOTAGE")
    sabotage_model = sabotage_auditor()
    with open("post_sabotage_log.jsonl") as f:
        post_sabotage_logs = [json.loads(line) for line in f]
    verify(prover_logs[CP_STEP:TOT_STEP], post_sabotage_logs,
           hash_model(prover_model), hash_model(sabotage_model),
           label="POST-TRAINING SABOTAGE AUDIT")

    # Test 4: Tampered checkpoint file -> rejected before deserialization
    print("\n Scenario 5: MODIFIED CHECKPOINT FILE (BROKEN SEAL)")
    _, secure = broken_seal_auditor()

    print("\n" + "=" * 72)
    print(f" CLEAN AUDIT: {'PASS' if clean_ok else 'FAIL'} | "
          f"BROKEN-SEAL REJECTED BEFORE LOAD: {'YES' if secure else 'NO'}")
    print("=" * 72)
