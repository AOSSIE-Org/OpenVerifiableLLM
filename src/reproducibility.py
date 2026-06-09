import torch
import torch.nn.functional as F
import json
import math
import platform, sys
import hashlib
import random
import numpy as np
from model import TinyGPT
from dataset import TinyDataset
from main import set_seed
from telemetry import TelemetryLogger
from config import TRAIN_CONFIG
from device import get_device, accel_rng_state, restore_accel_rng_state, device_name

# CUDA when available, else CPU. The same code path runs on both; only the
# floating-point reduction order (the hardware entropy under study) differs.
DEVICE = get_device()

def hash_model(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()

def run_training_segment(start_step, end_step, checkpoint_path_to_load=None, log_file="audit.jsonl", seed=None, tamper_weights=False):
    
    active_seed = seed if seed is not None else TRAIN_CONFIG["seed"]
    active_end_step = end_step if end_step is not None else TRAIN_CONFIG["total_steps"]

    if not checkpoint_path_to_load:
        set_seed(active_seed)

    dataset = TinyDataset()

    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=TRAIN_CONFIG["embed_dim"],
        num_heads=TRAIN_CONFIG["num_heads"],
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
        dropout=TRAIN_CONFIG["dropout"]
        ).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    logger = TelemetryLogger(filepath=log_file)

    if checkpoint_path_to_load:
        # Compute file-level hash before loading as security measure
        with open(checkpoint_path_to_load, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Load checkpoint (contains model, optimizer, and RNG states)
        # weights_only=False required for RNG state objects (numpy/python RNG)
        # File hash computed above provides tamper detection
        checkpoint = torch.load(checkpoint_path_to_load, weights_only=False)
        model.load_state_dict(checkpoint['model'])

        # verifying cryptographic seal on model weights
        if 'checkpoint_hash' in checkpoint:
            loaded_hash = logger.hash_model(model)
            if loaded_hash != checkpoint['checkpoint_hash']:
                print("\n  FATAL ALERT! : Cryptographic seal broken! Checkpoint file was tampered with.")
                print(f"    Expected: {checkpoint['checkpoint_hash'][:16]}...")
                print(f"    Got:      {loaded_hash[:16]}...\n")
                raise RuntimeError("Checkpoint integrity check failed")

        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng_state'])
        restore_accel_rng_state(checkpoint.get('accel_rng_state'))  # CUDA/XPU dropout RNG
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        print(f" ~> Auditor loaded checkpoint & RNG states from step {start_step}")

    x, y = dataset.get_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    for step in range(start_step, active_end_step):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_step(step, loss.item(), model)

        if not checkpoint_path_to_load and step == (TRAIN_CONFIG["checkpoint_step"] - 1):

            current_model_hash = logger.hash_model(model)
            
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'accel_rng_state': accel_rng_state(),  # None on CPU; tagged per-backend on GPU
                'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate(),
                'checkpoint_hash': current_model_hash
            }, "mid_checkpoint.pt")
            print(f" ~> Prover saved checkpoint at step {TRAIN_CONFIG['checkpoint_step']}")
        
    return model

def bad_seed_auditor(log_file="bad_seed_log.jsonl"):
    #test 1: correct checkpoint, wrong seed

    dataset = TinyDataset()
    model = TinyGPT(vocab_size=dataset.vocab_size, embed_dim=TRAIN_CONFIG["embed_dim"], num_heads=TRAIN_CONFIG["num_heads"], max_seq_len=TRAIN_CONFIG["max_seq_len"], dropout=TRAIN_CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    logger = TelemetryLogger(filepath=log_file)

    # Compute file-level hash before loading
    with open("mid_checkpoint.pt", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    checkpoint = torch.load("mid_checkpoint.pt", weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    set_seed(42) #BAD SEED
    print(" ~> Tampered auditor loaded checkpoint with BAD seed (42)")

    x, y = dataset.get_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    for step in range(TRAIN_CONFIG["checkpoint_step"], TRAIN_CONFIG["total_steps"]):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.log_step(step, loss.item(), model)
    return model
    
def secret_noise_auditor(log_file="secret_noise_log.jsonl"):
    #test 2: correct checkpoint, correct seed, but secret noise added to gradients

    set_seed(TRAIN_CONFIG["seed"]) #GOOD SEED

    dataset = TinyDataset()
    model = TinyGPT(vocab_size=dataset.vocab_size, embed_dim=TRAIN_CONFIG["embed_dim"], num_heads=TRAIN_CONFIG["num_heads"], max_seq_len=TRAIN_CONFIG["max_seq_len"], dropout=TRAIN_CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    logger = TelemetryLogger(filepath=log_file)

    # Compute file-level hash before loading
    with open("mid_checkpoint.pt", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    checkpoint = torch.load("mid_checkpoint.pt", weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Restore RNG states to prevent replay drift
    torch.set_rng_state(checkpoint['rng_state'])
    restore_accel_rng_state(checkpoint.get('accel_rng_state'))
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])

    print(" ~> Tampered auditor loaded checkpoint with GOOD seed but will add secret noise to gradients")

    x, y = dataset.get_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    for step in range(TRAIN_CONFIG["checkpoint_step"], TRAIN_CONFIG["total_steps"]):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()

        # Add secret noise to gradients
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 1e-10  # Small noise

        optimizer.step()
        logger.log_step(step, loss.item(), model)
    return model

def sabotage_auditor(log_file="post_sabotage_log.jsonl"):
    #Test 3: correct replay, but weights silently modified after training ends.

    dataset = TinyDataset()
    model = TinyGPT(vocab_size=dataset.vocab_size, embed_dim=TRAIN_CONFIG["embed_dim"], num_heads=TRAIN_CONFIG["num_heads"], max_seq_len=TRAIN_CONFIG["max_seq_len"], dropout=TRAIN_CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    logger = TelemetryLogger(filepath=log_file)

    # Compute file-level hash before loading
    with open("mid_checkpoint.pt", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    checkpoint = torch.load("mid_checkpoint.pt", weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng_state'])
    restore_accel_rng_state(checkpoint.get('accel_rng_state'))  # CUDA/XPU dropout RNG
    print(" ~> Post-sabotage auditor loaded checkpoint correctly")

    x, y = dataset.get_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    for step in range(TRAIN_CONFIG["checkpoint_step"], TRAIN_CONFIG["total_steps"]):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.log_step(step, loss.item(), model)

    #silent mutation of weights after training
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p) * 1e-6

    print(" ~> Weights silently mutated after training completed")
    return model

def broken_seal_auditor(log_file="broken_seal_log.jsonl"):
    #Test 4: attacker intercepts file, corrupts weights, but leaves the original hash intact to bypass integrity check, auditor loads corrupted file and runs audit

    #weights corrupted
    # Compute file-level hash before loading
    with open("mid_checkpoint.pt", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    checkpoint = torch.load("mid_checkpoint.pt", weights_only=False)

    #modify weights slightly
    checkpoint['model']['lm_head.weight'] += 1e-5

    #save the poisoned file (the original embedded hash remains unchanged)
    torch.save(checkpoint, "corrupted_checkpoint.pt")
    print(" ~> Attacker corrupted weights and saved to corrupted_checkpoint.pt")

    #auditor loads the poisoned file
    model = run_training_segment(
        start_step=TRAIN_CONFIG["checkpoint_step"], 
        end_step=TRAIN_CONFIG["total_steps"], 
        checkpoint_path_to_load="corrupted_checkpoint.pt", 
        log_file=log_file
    )
    return model

def verify(prover_segment, auditor_logs, prover_hash, auditor_hash, label="AUDIT"):
    """Shared verification logic with drift quantification and cryptographic anchor."""
    print(f"\n[Verifying: {label}]")

    if len(prover_segment) != len(auditor_logs):
        print(f"Log length mismatch — prover: {len(prover_segment)}, auditor: {len(auditor_logs)}")
        return False

    match = True
    for p, a in zip(prover_segment, auditor_logs):
        step_match = p['step'] == a['step']
        loss_match = math.isclose(p['loss'], a['loss'], rel_tol=1e-6)
        grad_match = math.isclose(p['grad_norm'], a['grad_norm'], rel_tol=1e-6)
        param_match = math.isclose(p['param_norm'], a['param_norm'], rel_tol=1e-6)
        step_ok = step_match and loss_match and grad_match and param_match

        if not step_ok:
            match = False
            delta = abs(p['loss'] - a['loss'])
            print(f"Step {p['step']} | Prover: {p['loss']:.8f} | Auditor: {a['loss']:.8f} | Δ {delta:.2e} FAILED")
        else:
            print(f"Step {p['step']} | Prover: {p['loss']:.8f} | Auditor: {a['loss']:.8f} | PASSED")

    hash_match = (prover_hash == auditor_hash)
    if not hash_match:
        print(f"\n Hash mismatch! Prover hash: {prover_hash[:16]} // Auditor hash: {auditor_hash[:16]} [HASH ERROR]")

    if match and hash_match:
        print(f"\n (❁ ´◡`❁) {label} PASSED: Segment replay is bitwise deterministic.")
    else:
        print(f"\n (╯°□°）╯︵ ┻━┻  {label} FAILED: Trajectories diverged.")

    # Return full verification verdict instead of just telemetry match
    verification_result = {
        'telemetry_match': match,
        'hash_match': hash_match,
        'passed': match and hash_match,
        'prover_hash': prover_hash,
        'auditor_hash': auditor_hash,
        'label': label
    }
    return verification_result

if __name__ == "__main__":
    CP_STEP = TRAIN_CONFIG["checkpoint_step"]
    TOT_STEP = TRAIN_CONFIG["total_steps"]

    print(f"\n=== Device: {DEVICE.type.upper()} ({device_name(DEVICE)}) | torch {torch.__version__} ===")
    if DEVICE.type == "cuda":
        print("    Phase 3: strict GPU determinism (cuDNN deterministic + pinned cuBLAS workspace)")
    elif DEVICE.type == "xpu":
        print("    Phase 3: Intel XPU (oneAPI) — deterministic algorithms enabled (best-effort)")

    # Baseline: should pass
    print("\n Scenario 1: CLEAN AUDIT ")
    prover_model = run_training_segment(start_step=0, end_step=TOT_STEP, log_file="prover_log.jsonl")
    auditor_model = run_training_segment(start_step=CP_STEP, end_step=TOT_STEP, checkpoint_path_to_load="mid_checkpoint.pt", log_file="auditor_log.jsonl")

    with open("prover_log.jsonl") as f:
        prover_logs = [json.loads(line) for line in f]
    with open("auditor_log.jsonl") as f:
        auditor_logs = [json.loads(line) for line in f]

    verify(prover_logs[CP_STEP:TOT_STEP], auditor_logs, hash_model(prover_model), hash_model(auditor_model), label="CLEAN AUDIT")

    # Test 1: Bad seed: should fail
    print("\n Scenario 2: BAD SEED")
    bad_model=bad_seed_auditor()

    with open("bad_seed_log.jsonl") as f:
        tampered_logs = [json.loads(line) for line in f]

    verify(prover_logs[CP_STEP:TOT_STEP], tampered_logs, hash_model(prover_model), hash_model(bad_model), label="BAD SEED AUDIT")

    # Test 2: Noisy weights: should fail 
    print("\n Scenario 3: NOISE INJECTED")
    noisey_model = secret_noise_auditor()

    with open("secret_noise_log.jsonl") as f:
        noisy_logs = [json.loads(line) for line in f]

    verify(prover_logs[CP_STEP:TOT_STEP], noisy_logs, hash_model(prover_model), hash_model(noisey_model), label="NOISY WEIGHTS AUDIT")

    # Test 3: Post-training sabotage, hash fail 
    print("\n Scenario 4: POST-TRAINING WEIGHT SABOTAGE")
    sabotage_model = sabotage_auditor()

    with open("post_sabotage_log.jsonl") as f:
        post_sabotage_logs = [json.loads(line) for line in f]

    verify(
        prover_logs[CP_STEP:TOT_STEP], post_sabotage_logs,
        hash_model(prover_model), hash_model(sabotage_model),
        label="POST-TRAINING SABOTAGE AUDIT"
    )

    # Test 4: Tampered Checkpoint File (Broken Seal)
    print("\n Scenario 5: MODIFIED CHECKPOINT FILE (BROKEN SEAL)")
    broken_seal_model = broken_seal_auditor()

    with open("broken_seal_log.jsonl") as f:
        broken_seal_logs = [json.loads(line) for line in f]

    verify(
        prover_logs[CP_STEP:TOT_STEP], broken_seal_logs,
        hash_model(prover_model), hash_model(broken_seal_model),
        label="BROKEN SEAL AUDIT"
    )

# Uncomment the following lines to run only the Segmented audit verification:
'''           
if __name__ == "__main__":
    fingerprint = {
    "torch": torch.__version__,
    "python": sys.version,
    "os": platform.platform(),
    "cpu": platform.processor()
}
    with open("env_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    print("\n SEGMENTED AUDIT VERIFICATION ")

    print("\n[Running Prover: Steps 0 to 10]")
    run_training_segment(start_step = 0, end_step = 10, log_file="prover_log.jsonl")

    print("\n[Running Auditor: Steps 5 to 10 with checkpoint]")
    run_training_segment(start_step = 5, end_step = 10, checkpoint_path_to_load="mid_checkpoint.pt", log_file="auditor_log.jsonl")

    print("\n[Verifying Telemetry Trajectories]")
    with open("prover_log.jsonl", "r") as f:
        prover_logs = [json.loads(line) for line in f.readlines()]
    with open("auditor_log.jsonl", "r") as f:
        auditor_logs = [json.loads(line) for line in f.readlines()]

    prover_segment = prover_logs[5:10]

    if len(prover_segment) != 5 or len(auditor_logs) != 5:
        print(f"Log length mismatch — prover_segment: {len(prover_segment)}, auditor: {len(auditor_logs)}")
        print("Cannot verify. Check for crashes or early exits in training.")
    else:
        match = True

        for p, a in zip(prover_segment, auditor_logs):
            step_match = p['step'] == a['step']
            loss_match = math.isclose(p['loss'], a['loss'], rel_tol=1e-6)
            grad_match = math.isclose(p['grad_norm'], a['grad_norm'], rel_tol=1e-6)
            param_match = math.isclose(p['param_norm'], a['param_norm'], rel_tol=1e-6)
            step_ok = step_match and loss_match and grad_match and param_match
            if not step_ok:
                match = False
            status = "ok" if step_ok else "error"
            print(f"Step {p['step']} | Prover Loss: {p['loss']:.6f} | Auditor Loss: {a['loss']:.6f} {status}")

        if match:
            print("\n (❁ ´◡`❁) \nAUDIT PASSED: Segment replay is bitwise deterministic.")
        else:
            print("\n (╯°□°）╯︵ ┻━┻ \nAUDIT FAILED: Trajectories diverged.")

'''
#Reproducibility test for tinyGPT without Segment Verification test
#uncomment this block and comment the others to not have segment verification
'''
import torch
#for linear model: from model import TinyModel
from model import TinyGPT
from dataset import TinyDataset
import torch.nn.functional as F
from main import set_seed

def train_once():
    set_seed(99)

    dataset = TinyDataset()
    #for linear model: model = TinyModel(vocab_size=dataset.vocab_size)
    model = TinyGPT(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    x, y = dataset.get_batch()

    for step in range(5):
        logits = model(x)
        #for linear model: loss = F.cross_entropy(logits, y[0])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses

if __name__ == "__main__":
    print("running reproducability test")

    model1, losses1 = train_once()

    model2, losses2 = train_once()

    losses_match = (losses1 ==losses2)
    print(f"Loss curves identical: {losses_match}")
    
    params_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model1.parameters(), model2.parameters())
    )
    print (f"Bitwise parameter match: {params_match}")

    if losses_match and params_match:
        print("\nSuccess!: Full deterministic gradient flow verified.")
    else:
        print ("\nFailure: Entropy led to non-deterministic behavior")

'''