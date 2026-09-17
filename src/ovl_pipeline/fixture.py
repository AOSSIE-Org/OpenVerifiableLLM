"""Synthetic end-to-end development fixture, explicitly not production evidence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile

from nacl.signing import SigningKey
from safetensors import safe_open
from safetensors.torch import load_file, save_file, save
from tokenizers import Tokenizer
import torch

from .canonical import EvidenceError, canonical, confined, read_json, digest, file_hash, inventory, parse_json, sha256, verify_inventory, write_json
from .data import OFFSET, PAD, BOS, EOS, USER, ASSISTANT, extract_wikipedia, prepare_stream, rows, text_ids, train_tokenizer, wikipedia_documents
from .state import read_state, restore, tensor_digest
from .training import TrustPolicy, code_root, full_replay, initialize, make_registration, signed, train, verify_signed


def prepare(raw, output):
    output.mkdir(parents=True, exist_ok=False)
    corpus = extract_wikipedia([raw / "wiki.xml"], output / "corpus")
    tokenizer = train_tokenizer(output / "corpus/articles.jsonl", output / "tokenizer", vocab_size=320, sample_bytes=100000)
    tp = output / "tokenizer/tokenizer.json"
    wiki = prepare_stream(wikipedia_documents(output / "corpus/articles.jsonl"), tp, output / "wikipedia", phase="wikipedia")
    chat = prepare_stream(read_json(confined(raw, "conversations.json"), canonical_required=False), tp, output / "conversation", phase="conversation")
    return {"corpus": corpus, "tokenizer": tokenizer, "streams": {"wikipedia": wiki, "conversation": chat}}


def recipe(vocab_size):
    return {"seed": 1234, "context": 16, "batch_size": 3, "boundary_every": 3,
            "learning_rate": "0.0003", "weight_decay": "0.01", "init": "normal-0.02-tied-v1",
            "model": {"vocab_size": vocab_size, "embed_dim": 16, "num_heads": 2, "num_layers": 1,
                      "max_seq_len": 16, "dropout": 0, "attn_impl": "manual"}}


def export_config(registration, phase, model_root, tokenizer_dir):
    return {"schema": "ovl.export.v1", "recipe": registration["recipe"],
            "aliases": {"lm_head.weight": "transformer.wte.weight"}, "model_root": model_root,
            "loader_code_root": registration["code_root"],
            "tokenizer": {"file": "tokenizer.json", "sha256": file_hash(tokenizer_dir / "tokenizer.json"),
                          "manifest_file": "tokenizer-manifest.json", "manifest_sha256": file_hash(tokenizer_dir / "tokenizer-manifest.json"),
                          "text_id_offset": OFFSET, "controls": {"pad": PAD, "bos": BOS, "eos": EOS, "user": USER, "assistant": ASSISTANT}},
            "inference": {"input_format": "bos-text-v1" if phase == "base" else "bos-user-text-eos-assistant-v1",
                          "decoding": "greedy", "max_new_tokens": 6}}


def export_model(registration, boundary, training_dir, output, tokenizer_dir, phase):
    output.mkdir(parents=True, exist_ok=False)
    model, opt, _ = initialize(registration["recipe"])
    from .canonical import confined
    md, tensors = read_state(confined(training_dir, boundary["checkpoint_path"]), boundary["checkpoint"])
    restore(model, opt, md, tensors)
    state = dict(model.state_dict())
    root = tensor_digest(state)
    del state["lm_head.weight"]
    save_file({k: v.contiguous().clone() for k, v in state.items()}, str(output / "model.safetensors"))
    for name in ("tokenizer.json", "tokenizer-manifest.json"):
        shutil.copyfile(tokenizer_dir / name, output / name)
    write_json(output / "config.json", export_config(registration, phase, root, tokenizer_dir))
    return root


def load_export(directory, *, expected_recipe=None):
    config = read_json(confined(directory, "config.json"))
    if set(config) != {"schema", "recipe", "aliases", "model_root", "loader_code_root", "tokenizer", "inference"} or config.get("schema") != "ovl.export.v1" or config["aliases"] != {"lm_head.weight": "transformer.wte.weight"}:
        raise EvidenceError("unsupported model export")
    if expected_recipe is not None and config["recipe"] != expected_recipe:
        raise EvidenceError("export recipe differs from registration")
    if config["loader_code_root"] != code_root():
        raise EvidenceError("export requires a different trusted loader source")
    token = config["tokenizer"]
    if token["file"] != "tokenizer.json" or token["manifest_file"] != "tokenizer-manifest.json" or token["text_id_offset"] != OFFSET:
        raise EvidenceError("unsupported export tokenizer recipe")
    if file_hash(confined(directory, token["file"])) != token["sha256"] or file_hash(confined(directory, token["manifest_file"])) != token["manifest_sha256"]:
        raise EvidenceError("export tokenizer hash mismatch")
    model, _, _ = initialize(config["recipe"])
    weight_path = confined(directory, "model.safetensors")
    with safe_open(str(weight_path), framework="pt", device="cpu") as f:
        if f.metadata():
            raise EvidenceError("unexpected export metadata")
    state = load_file(str(weight_path))
    expected = dict(model.state_dict())
    del expected["lm_head.weight"]
    if set(state) != set(expected) or any(state[k].dtype != expected[k].dtype or state[k].shape != expected[k].shape for k in expected):
        raise EvidenceError("export tensor names, dtype or shape mismatch")
    if sha256(save(state)) != file_hash(weight_path):
        raise EvidenceError("export bytes differ from the canonical serialization recipe")
    state["lm_head.weight"] = state["transformer.wte.weight"]
    model.load_state_dict(state, strict=True)
    if tensor_digest(dict(model.state_dict())) != config["model_root"]:
        raise EvidenceError("export canonical state mismatch")
    return model


def infer(directory, tokenizer_path=None, prompt="Alpha", steps=6):
    model = load_export(directory)
    model.eval()
    config = read_json(confined(directory, "config.json"))
    if tokenizer_path is not None and file_hash(tokenizer_path) != config["tokenizer"]["sha256"]:
        raise EvidenceError("caller tokenizer differs from exported tokenizer")
    tokenizer = Tokenizer.from_file(str(confined(directory, "tokenizer.json")))
    formatter = config["inference"]["input_format"]
    if formatter == "bos-text-v1":
        tokens = [BOS] + text_ids(tokenizer, prompt)
    elif formatter == "bos-user-text-eos-assistant-v1":
        tokens = [BOS, USER] + text_ids(tokenizer, prompt) + [EOS, ASSISTANT]
    else:
        raise EvidenceError("unsupported inference template")
    input_ids = tokens.copy()
    generated = []
    context = model.transformer["wpe"].num_embeddings
    with torch.no_grad():
        for _ in range(steps):
            value = int(model(torch.tensor([tokens[-context:]], dtype=torch.int64))[0, -1].argmax())
            tokens.append(value)
            generated.append(value)
            if value == EOS:
                break
    return {"prompt": prompt, "input_ids": input_ids, "input_format": formatter, "output_ids": generated,
            "decoding": "greedy", "max_new_tokens": steps}


def run_fixture(source, output, policy_path):
    output = Path(output)
    policy_path = Path(policy_path)
    if policy_path.resolve().is_relative_to(output.resolve()):
        raise EvidenceError("trust policy must be supplied separately from model bundle")
    if policy_path.exists():
        raise EvidenceError("refuse to overwrite trust policy")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    for name in ("wiki.xml", "conversations.json"):
        shutil.copyfile(source / name, raw / name)
    raw_inventory = inventory(raw, ["wiki.xml", "conversations.json"])
    write_json(output / "source.json", {"schema": "ovl.source.v1", "scope": "synthetic", "files": raw_inventory})
    prepared = prepare(raw, output / "prepared")
    write_json(output / "preparation.json", prepared)
    # Ephemeral fixture-only private key stays in memory and is never published.
    key = SigningKey.generate()
    registration = make_registration(recipe(prepared["tokenizer"]["vocab_size"]), prepared["streams"], key,
                                     digest(raw_inventory), digest(prepared))
    policy = TrustPolicy(digest(registration), bytes(key.verify_key).hex(), "local-synthetic-fixture")
    write_json(policy_path, asdict(policy))
    write_json(output / "registration.json", signed(registration, key))
    dirs = {p: output / "prepared" / p for p in prepared["streams"]}
    chain = train(registration, policy, dirs, output / "training", key)
    # Fresh reconstruction is exercised by verify_fixture below. This first report
    # also establishes export equality from the continuously replayed model.
    report = full_replay(registration, policy, dirs, output / "training")
    write_json(output / "replay.json", report)
    for phase, kind in (("base", "base"), ("chat", "final")):
        b = next(b["body"] for b in chain if b["body"]["kind"] == kind)
        if export_model(registration, b, output / "training", output / phase, output / "prepared/tokenizer", phase) != report[phase + "_model_root"]:
            raise EvidenceError("export differs from full replay")
    for phase in ("base", "chat"):
        write_json(output / phase / "inference.json", infer(output / phase, output / "prepared/tokenizer/tokenizer.json"))
    names = [p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()]
    release = {"schema": "ovl.fixture-release.v1", "scope": policy.scope,
               "registration": policy.registration_sha256, "chain_root": report["chain_root"],
               "files": inventory(output, names)}
    write_json(output / "release.json", signed(release, key))
    # Verify a clean copy as a local stand-in for download. Never call this a public download.
    with tempfile.TemporaryDirectory(prefix="ovl-clean-copy-") as tmp:
        clean = Path(tmp) / "bundle"
        shutil.copytree(output, clean)
        verified = verify_fixture(clean, policy)
    return {**verified, "release_root": digest(release), "trust_policy": str(policy_path),
            "transport_scope": "local-copy-not-public-download"}


def verify_fixture(bundle, policy):
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise EvidenceError("symlink in fixture bundle")
    envelope = read_json(confined(bundle, "registration.json"))
    registration = verify_signed(envelope, policy.run_public_key_hex)
    policy.validate(registration)
    release = verify_signed(read_json(confined(bundle, "release.json")), policy.run_public_key_hex)
    if release.get("schema") != "ovl.fixture-release.v1" or release.get("scope") != policy.scope or release["registration"] != policy.registration_sha256:
        raise EvidenceError("release ancestry/scope mismatch")
    verify_inventory(bundle, release["files"])
    actual_names = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {"release.json"}
    if actual_names != {e["path"] for e in release["files"]}:
        raise EvidenceError("unlisted or missing release file")
    source = read_json(confined(bundle, "source.json"))
    if source.get("schema") != "ovl.source.v1" or source.get("scope") != "synthetic" or {e["path"] for e in source["files"]} != {"wiki.xml", "conversations.json"}:
        raise EvidenceError("invalid fixture source inventory")
    raw_dir = confined(bundle, "raw")
    training_dir = confined(bundle, "training")
    verify_inventory(raw_dir, source["files"])
    if digest(source["files"]) != registration["raw_root"]:
        raise EvidenceError("raw inventory differs from registration")
    with tempfile.TemporaryDirectory(prefix="ovl-reconstruction-") as tmp:
        rebuilt_dir = Path(tmp) / "prepared"
        rebuilt = prepare(raw_dir, rebuilt_dir)
        if digest(rebuilt) != registration["preparation_root"] or rebuilt["streams"] != registration["streams"]:
            raise EvidenceError("raw-to-prepared reconstruction mismatch")
        if read_json(confined(bundle, "preparation.json")) != rebuilt:
            raise EvidenceError("published preparation manifest differs from reconstruction")
        rebuilt_names = sorted(p.relative_to(rebuilt_dir).as_posix() for p in rebuilt_dir.rglob("*") if p.is_file())
        published_dir = confined(bundle, "prepared")
        published_names = sorted(p.relative_to(published_dir).as_posix() for p in published_dir.rglob("*") if p.is_file())
        if published_names != rebuilt_names or inventory(published_dir, published_names) != inventory(rebuilt_dir, rebuilt_names):
            raise EvidenceError("published prepared artifacts differ from reconstruction")
        report = full_replay(registration, policy, {p: rebuilt_dir / p for p in rebuilt["streams"]}, training_dir)
        if release["chain_root"] != report["chain_root"]:
            raise EvidenceError("release chain differs from replay")
        if read_json(confined(bundle, "replay.json")) != report:
            raise EvidenceError("stored replay attestation differs from local recomputation")
        chain = read_json(confined(training_dir, "chain.json"))
        expected_names = {"registration.json", "source.json", "preparation.json", "replay.json", "training/chain.json"}
        expected_names.update("raw/" + e["path"] for e in source["files"])
        expected_names.update("prepared/" + name for name in rebuilt_names)
        for envelope in chain["boundaries"]:
            path = envelope["body"]["checkpoint_path"]
            expected_names.update("training/" + path + "/" + name for name in ("state.json", "state.safetensors", "checkpoint.json"))
        for phase in ("base", "chat"):
            expected_names.update(phase + "/" + name for name in ("model.safetensors", "config.json", "inference.json", "tokenizer.json", "tokenizer-manifest.json"))
        if actual_names != expected_names:
            raise EvidenceError("unexpected release file set")
        for phase in ("base", "chat"):
            expected_config = export_config(registration, phase, report[phase + "_model_root"], rebuilt_dir / "tokenizer")
            if read_json(confined(bundle, phase + "/config.json")) != expected_config:
                raise EvidenceError("published inference/export config differs from registered recipe")
            model = load_export(bundle / phase, expected_recipe=registration["recipe"])
            if tensor_digest(dict(model.state_dict())) != report[phase + "_model_root"]:
                raise EvidenceError("export does not match replayed weights")
            receipt = read_json(confined(bundle, phase + "/inference.json"))
            if receipt != infer(bundle / phase, rebuilt_dir / "tokenizer/tokenizer.json"):
                raise EvidenceError("inference receipt mismatch")
    return {"schema": "ovl.fixture-verification.v1", "result": "PASS", "profile": policy.scope,
            "raw_reconstruction": "PASS", "continuous_replay": "PASS", "export_inference": "PASS",
            "public_anchoring": "NOT_RUN", "production_acceptance": "NOT_RUN",
            "updates": report["updates"], "boundaries": len(report["boundaries"]),
            "locally_recomputed": True, "performed_by": "project-operator", "independent_third_party": False}
