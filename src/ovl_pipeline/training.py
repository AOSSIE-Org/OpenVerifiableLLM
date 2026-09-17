"""Shared training/replay kernel. Production admission is intentionally closed in P0."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import shutil
import struct
import uuid

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
import numpy as np
import torch
from torch.nn import functional as F

from model import TinyGPT
from .canonical import EvidenceError, canonical, confined, digest, file_hash, inventory, read_json, sha256, write_json
from . import schema
from .data import batches, check_coverage, rows, validate_stream
from .state import capture, read_state, restore, save_state, state_root, tensor_digest


MAX_BOUNDARIES = 4096


@dataclass(frozen=True)
class TrustPolicy:
    registration_sha256: str
    run_public_key_hex: str
    scope: str

    def validate(self, registration):
        schema.registration(registration)
        if self.scope != "local-synthetic-fixture":
            raise EvidenceError("UNAVAILABLE: production transparency/identity admission not implemented")
        if registration.get("schema") != "ovl.registration.v1" or registration.get("scope") != self.scope:
            raise EvidenceError("registration scope/schema mismatch")
        if digest(registration) != self.registration_sha256:
            raise EvidenceError("registration differs from externally configured trust root")
        if registration.get("run_public_key") != self.run_public_key_hex:
            raise EvidenceError("run key not authorized by external policy")


def signed(body, key):
    return {"body": body, "signature": key.sign(canonical(body)).signature.hex()}


def verify_signed(envelope, public_key):
    if type(envelope) is not dict or set(envelope) != {"body", "signature"}:
        raise EvidenceError("invalid signed envelope")
    try:
        VerifyKey(bytes.fromhex(public_key)).verify(canonical(envelope["body"]), bytes.fromhex(envelope["signature"]))
    except (ValueError, BadSignatureError) as e:
        raise EvidenceError("run signature mismatch") from e
    return envelope["body"]


def configure(seed):
    # Initialization arithmetic remains on CPU. The explicit GPU wrapper must
    # configure CUDA first; production admission remains a separate closed gate.
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment():
    packages = {n: importlib.metadata.version(n) for n in
                ("torch", "numpy", "safetensors", "tokenizers", "mwparserfromhell", "rfc8785", "defusedxml", "pynacl")}
    return {"schema": "ovl.environment.v1", "python": platform.python_version(),
            "machine": platform.machine(), "system": platform.system(), "packages": packages,
            "device": "cpu", "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"), "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_build": torch.__config__.show()}


def code_root():
    base = Path(__file__).parent
    return digest([{ "path": "ovl_pipeline/" + p.name, "sha256": file_hash(p)} for p in sorted(base.glob("*.py"))]
                  + [{"path": "model.py", "sha256": file_hash(base.parent / "model.py")}])


def new_optimizer(model, recipe):
    return torch.optim.AdamW(model.parameters(), lr=float(recipe["learning_rate"]),
                             weight_decay=float(recipe["weight_decay"]), betas=(0.9, 0.999), eps=1e-8,
                             amsgrad=False, maximize=False, differentiable=False,
                             foreach=False, fused=False, capturable=False)


def initialize(recipe, *, device="cpu"):
    if device not in ("cpu", "cuda:0"):
        raise EvidenceError("unsupported kernel device")
    schema.recipe(recipe, gpu=device == "cuda:0")
    if device == "cuda:0" and (not torch.cuda.is_initialized() or torch.cuda.device_count() != 1):
        raise EvidenceError("GPU runtime must be explicitly configured before initialization")
    configure(recipe["seed"])
    cfg = recipe["model"]
    model = TinyGPT(**cfg)
    model.lm_head.weight = model.transformer["wte"].weight
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        elif name.endswith("weight"):
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.zeros_(p)
    model.train()
    if device == "cuda:0":
        # The GPU runtime must already have configured the sole visible device.
        model.to(device)
        model.lm_head.weight = model.transformer["wte"].weight
    optimizer = new_optimizer(model, recipe)
    control = {"phase": "wikipedia", "global_step": 0, "phase_step": 0, "cursor": 0,
               "transcript": sha256(b"ovl.batch-transcript.v1"), "schedule": "constant-lr-v1",
               "accumulation": "none", "scaler": "none"}
    return model, optimizer, control


def update(model, optimizer, batch, control, total, *, precision="fp32", metrics=None):
    device = next(model.parameters()).device
    if device.type not in ("cpu", "cuda") or precision not in ("fp32", "bf16"):
        raise EvidenceError("unsupported update device/precision")
    if precision == "bf16" and device.type != "cuda":
        raise EvidenceError("BF16 profile requires CUDA; no silent CPU fallback")
    if any(p.device != device or p.dtype != torch.float32 for p in model.parameters()):
        raise EvidenceError("kernel requires FP32 master parameters on one device")
    if any(t.device.type != "cpu" for t in batch.values()):
        raise EvidenceError("coverage and transcript require original CPU batch tensors")
    next_cursor = check_coverage(batch, control["cursor"], total)
    batch_root = tensor_digest(batch)
    inputs, targets, valid = (batch[n].to(device) for n in ("inputs", "targets", "mask"))
    context = torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if precision == "bf16" else nullcontext()
    with context:
        logits = model(inputs)
        losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
        valid = valid.flatten()
        loss = losses[valid].sum() / valid.sum()
    if not torch.isfinite(loss):
        raise EvidenceError("nonfinite loss")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise EvidenceError("nonfinite gradients")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise EvidenceError("nonfinite updated parameter")
    optimizer.zero_grad(set_to_none=True)
    if metrics is not None:
        metrics.update(loss_float64_hex=struct.pack(">d",loss.item()).hex(),
                       targets=next_cursor-control["cursor"])
    control = {**control, "global_step": control["global_step"] + 1, "phase_step": control["phase_step"] + 1,
               "cursor": next_cursor, "transcript": digest({"previous": control["transcript"], "batch": batch_root})}
    return control


def make_registration(recipe, streams, key, raw_root, preparation_root):
    model, opt, control = initialize(recipe)
    metadata, tensors = capture(model, opt, control)
    return {"schema": "ovl.registration.v1", "scope": "local-synthetic-fixture",
            "run_id": "synthetic-p0", "attempt_id": "local-v1", "recipe": recipe,
            "code_root": code_root(), "environment": environment(), "raw_root": raw_root,
            "preparation_root": preparation_root, "streams": streams,
            "initial_state": state_root(metadata, tensors), "run_public_key": bytes(key.verify_key).hex(),
            "anchoring": "NOT_RUN-local-fixture-only", "conversation_policy": "one-epoch-reset-adamw-v1"}


def validate_registration(registration, policy, stream_dirs):
    policy.validate(registration)
    configure(registration["recipe"]["seed"])
    if registration["code_root"] != code_root() or registration["environment"] != environment():
        raise EvidenceError("code/environment differs from registration")
    if set(registration["streams"]) != {"wikipedia", "conversation"} or set(stream_dirs) != set(registration["streams"]):
        raise EvidenceError("both complete phases required")
    for phase, manifest in registration["streams"].items():
        if manifest["phase"] != phase:
            raise EvidenceError("stream phase mismatch")
        validate_stream(stream_dirs[phase], manifest)
    # Conservative upper bound includes zero-loss chat windows. Reject too-large
    # JSON chains before any update instead of creating an unreplayable run.
    r = registration["recipe"]
    boundaries = 2  # initial state and base-to-chat transition
    for directory in stream_dirs.values():
        windows = sum((d["tokens"] + r["context"] - 1) // r["context"] for d in rows(directory / "documents.jsonl"))
        updates = (windows + r["batch_size"] - 1) // r["batch_size"]
        boundaries += (updates + r["boundary_every"] - 1) // r["boundary_every"]
    if boundaries > MAX_BOUNDARIES:
        raise EvidenceError("registered boundary schedule exceeds bounded chain format")


def _initial(registration):
    model, opt, control = initialize(registration["recipe"])
    if state_root(*capture(model, opt, control)) != registration["initial_state"]:
        raise EvidenceError("regenerated initialization mismatch")
    return model, opt, control


def _transition(model, recipe, control):
    return new_optimizer(model, recipe), {**control, "phase": "conversation", "phase_step": 0, "cursor": 0}


def validate_chain(registration, policy, envelopes, *, complete):
    policy.validate(registration)
    if not envelopes or len(envelopes) > MAX_BOUNDARIES:
        raise EvidenceError("empty or oversized boundary chain")
    previous, step = policy.registration_sha256, -1
    previous_control = None
    for i, env in enumerate(envelopes):
        b = verify_signed(env, policy.run_public_key_hex)
        if b.get("schema") != "ovl.boundary.v1" or b["registration"] != policy.registration_sha256 or b["previous"] != previous or b["index"] != i:
            raise EvidenceError("broken boundary ancestry/order")
        schema.fields(b, "schema index registration previous kind control checkpoint_path checkpoint", "boundary")
        c = b["control"]
        schema.control(c)
        if b["checkpoint_path"] != f"boundary-{i:05d}":
            raise EvidenceError("unexpected checkpoint path")
        if c["global_step"] < step or (c["global_step"] == step and b["kind"] != "transition"):
            raise EvidenceError("boundary step order")
        if i == 0 and (b["kind"] != "initial" or c["global_step"] != 0):
            raise EvidenceError("missing initial boundary")
        if c["phase"] not in registration["streams"]:
            raise EvidenceError("unknown boundary phase")
        if not 0 <= c["cursor"] <= registration["streams"][c["phase"]]["targets"]:
            raise EvidenceError("invalid cursor")
        if previous_control and previous_control["phase"] == "conversation" and c["phase"] == "wikipedia":
            raise EvidenceError("backward phase transition")
        previous_control = c
        step, previous = c["global_step"], digest(b)
    if complete:
        final = envelopes[-1]["body"]
        if final["kind"] != "final" or final["control"]["phase"] != "conversation" or final["control"]["cursor"] != registration["streams"]["conversation"]["targets"]:
            raise EvidenceError("incomplete final boundary")
    return previous


def train(registration, policy, stream_dirs, output: Path, key: SigningKey, *, stop_after=None, resume=False,
          recovery_directory=None):
    validate_registration(registration, policy, stream_dirs)
    if bytes(key.verify_key).hex() != policy.run_public_key_hex:
        raise EvidenceError("wrong signing key")
    recipe = registration["recipe"]
    model, opt, control = _initial(registration)
    if resume:
        chain_path = confined(output, "chain.json")
        if chain_path.exists():
            log = read_json(chain_path)
            envelopes = log["boundaries"]
            validate_chain(registration, policy, envelopes, complete=log["complete"])
            if log["complete"]:
                raise EvidenceError("cannot resume a completed chain")
            last = envelopes[-1]["body"]
            if last["control"]["phase"] == "conversation":
                opt = new_optimizer(model, recipe)
            md, tensors = read_state(confined(output, last["checkpoint_path"]), last["checkpoint"])
            control = restore(model, opt, md, tensors)
            if control != last["control"]:
                raise EvidenceError("checkpoint/control mismatch")
        else:
            if not output.is_dir() or any(p.name != "boundary-00000" for p in output.iterdir()):
                raise EvidenceError("unrecognized incomplete initial run")
            envelopes = []
    else:
        output.mkdir(parents=True, exist_ok=False)
        envelopes = []

    def boundary(kind):
        name = f"boundary-{len(envelopes):05d}"
        target = confined(output, name)
        if target.exists():
            marker = confined(target, "checkpoint.json")
            if marker.is_file():
                # A durable but uncommitted checkpoint can be reused only after
                # regeneration reaches the identical full state, without loading it.
                ck = read_json(marker)
                read_state(target, ck)
                if ck["state_root"] != state_root(*capture(model, opt, control)):
                    raise EvidenceError("orphan checkpoint differs from regenerated state")
            else:
                # Preserve incomplete evidence outside the canonical run; never
                # delete the only bytes or treat directory existence as completion.
                recovery = (confined(output.parent, output.name + "-recovery") if recovery_directory is None
                            else Path(recovery_directory))
                if recovery.resolve().is_relative_to(output.resolve()):
                    raise EvidenceError("recovery evidence must be outside the training directory")
                recovery.mkdir(parents=True, exist_ok=True)
                destination = recovery / (name + "-" + uuid.uuid4().hex)
                files = inventory(target, [p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()])
                shutil.move(str(target), str(destination))
                write_json(recovery / (destination.name + ".json"),
                           {"schema": "ovl.recovery-observation.v1", "scope": "operator-local discarded incomplete checkpoint",
                            "registration": policy.registration_sha256, "original_path": name,
                            "preserved_directory": destination.name, "files": files})
                ck = save_state(target, model, opt, control)
        else:
            ck = save_state(target, model, opt, control)
        body = {"schema": "ovl.boundary.v1", "index": len(envelopes), "registration": policy.registration_sha256,
                "previous": digest(envelopes[-1]["body"]) if envelopes else policy.registration_sha256,
                "kind": kind, "control": control.copy(), "checkpoint_path": name, "checkpoint": ck}
        envelopes.append(signed(body, key))
        write_json(output / "chain.json", {"schema": "ovl.chain.v1", "complete": kind == "final", "boundaries": envelopes})

    if not envelopes:
        boundary("initial")
    for phase in ("wikipedia", "conversation"):
        if phase == "wikipedia" and control["phase"] == "conversation":
            continue
        if phase == "conversation" and control["phase"] != phase:
            opt, control = _transition(model, recipe, control)
            boundary("transition")
        total = registration["streams"][phase]["targets"]
        opening_step = control["phase_step"]
        for index, batch in enumerate(batches(stream_dirs[phase], recipe["context"], recipe["batch_size"])):
            if index < opening_step:
                continue
            control = update(model, opt, batch, control, total)
            final_phase = control["cursor"] == total
            scheduled = control["phase_step"] % recipe["boundary_every"] == 0 or final_phase
            if scheduled:
                boundary("final" if final_phase and phase == "conversation" else "base" if final_phase else "progress")
            if stop_after is not None and control["global_step"] >= stop_after:
                # P0 test interruption occurs only at a registered boundary.
                if not scheduled:
                    continue
                return envelopes
        if control["cursor"] != total:
            raise EvidenceError("incomplete phase coverage")
    validate_chain(registration, policy, envelopes, complete=True)
    return envelopes


def full_replay(registration, policy, stream_dirs, training_dir):
    validate_registration(registration, policy, stream_dirs)
    log = read_json(confined(training_dir, "chain.json"))
    if log.get("schema") != "ovl.chain.v1" or log.get("complete") is not True:
        raise EvidenceError("full replay requires complete chain")
    envelopes = log["boundaries"]
    chain_root = validate_chain(registration, policy, envelopes, complete=True)
    recipe = registration["recipe"]
    model, opt, control = _initial(registration)
    compared, base_root = [], None
    position = 0

    def compare(kind):
        nonlocal position, base_root
        if position >= len(envelopes):
            raise EvidenceError("missing scheduled boundary")
        b = envelopes[position]["body"]
        if b["control"] != control or b["kind"] != kind:
            raise EvidenceError("boundary schedule/control mismatch")
        # Validate committed checkpoint bytes, but NEVER load them into replay state.
        from .canonical import confined
        md, tensors = read_state(confined(training_dir, b["checkpoint_path"]), b["checkpoint"])
        actual = state_root(*capture(model, opt, control))
        if actual != state_root(md, tensors):
            raise EvidenceError(f"continuous state mismatch at boundary {position}")
        compared.append({"index": position, "state_root": actual, "result": "PASS"})
        if kind == "base":
            base_root = tensor_digest(dict(model.state_dict()))
        position += 1

    compare("initial")
    for phase in ("wikipedia", "conversation"):
        if phase == "conversation":
            opt, control = _transition(model, recipe, control)
            compare("transition")
        total = registration["streams"][phase]["targets"]
        for batch in batches(stream_dirs[phase], recipe["context"], recipe["batch_size"]):
            control = update(model, opt, batch, control, total)
            end = control["cursor"] == total
            if control["phase_step"] % recipe["boundary_every"] == 0 or end:
                compare("final" if end and phase == "conversation" else "base" if end else "progress")
        if control["cursor"] != total:
            raise EvidenceError("incomplete replay coverage")
    if position != len(envelopes) or base_root is None:
        raise EvidenceError("extra boundaries or missing base")
    return {"schema": "ovl.replay.v1", "profile": "local-synthetic-continuous-replay",
            "result": "PASS", "public_end_to_end_verification": "NOT_RUN", "locally_recomputed": True,
            "performed_by": "project-operator", "independent_third_party": False,
            "registration": policy.registration_sha256, "chain_root": chain_root,
            "boundaries": compared, "base_model_root": base_root,
            "chat_model_root": tensor_digest(dict(model.state_dict())), "updates": control["global_step"]}
