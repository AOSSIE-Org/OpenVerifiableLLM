"""Closed v1 structural contracts for update-affecting configuration."""
import math
import re

from .canonical import EvidenceError, require_digest


def fields(value, names, label):
    if type(value) is not dict or set(value) != set(names.split()):
        raise EvidenceError(f"invalid {label} fields")


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise EvidenceError(f"invalid {label}")


def recipe(value):
    fields(value, "seed context batch_size boundary_every learning_rate weight_decay init model", "recipe")
    integer(value["seed"], 0, 2**32 - 1, "seed")
    integer(value["context"], 1, 32768, "context")
    integer(value["batch_size"], 1, 4096, "batch size")
    integer(value["boundary_every"], 1, 10**9, "boundary interval")
    for key, allow_zero in (("learning_rate", False), ("weight_decay", True)):
        s = value[key]
        if type(s) is not str or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", s):
            raise EvidenceError(f"invalid recipe {key}")
        n = float(s)
        if not math.isfinite(n) or n > 1 or (n < 0 if allow_zero else n <= 0):
            raise EvidenceError(f"invalid recipe {key}")
    if value["init"] != "normal-0.02-tied-v1":
        raise EvidenceError("unsupported recipe initialization")
    m = value["model"]
    fields(m, "vocab_size embed_dim num_heads num_layers max_seq_len dropout attn_impl", "model recipe")
    for key, low, high in (("vocab_size", 261, 65536), ("embed_dim", 1, 4096), ("num_heads", 1, 128),
                           ("num_layers", 1, 128), ("max_seq_len", 1, 32768)):
        integer(m[key], low, high, key)
    if m["embed_dim"] % m["num_heads"] or m["max_seq_len"] != value["context"]:
        raise EvidenceError("inconsistent model recipe dimensions")
    if type(m["dropout"]) is not int or m["dropout"] != 0 or m["attn_impl"] != "manual":
        raise EvidenceError("unsupported fixture dropout/attention recipe")
    # Bound configuration-directed allocation for this CPU-only verifier profile.
    d, layers, seq = m["embed_dim"], m["num_layers"], m["max_seq_len"]
    elements = (m["vocab_size"] + seq) * d + layers * (12*d*d + 13*d + seq*seq) + 2*d
    if elements > 100_000_000 or value["batch_size"] * seq * m["vocab_size"] > 100_000_000:
        raise EvidenceError("fixture recipe allocation budget exceeded")


def registration(value):
    fields(value, "schema scope run_id attempt_id recipe code_root environment raw_root preparation_root streams initial_state run_public_key anchoring conversation_policy", "registration")
    if value["schema"] != "ovl.registration.v1" or value["scope"] != "local-synthetic-fixture":
        raise EvidenceError("unsupported registration scope/schema")
    if value["anchoring"] != "NOT_RUN-local-fixture-only" or value["conversation_policy"] != "one-epoch-reset-adamw-v1":
        raise EvidenceError("unsupported fixture anchoring/phase policy")
    recipe(value["recipe"])
    for field in ("code_root", "raw_root", "preparation_root", "initial_state", "run_public_key"):
        require_digest(value[field])


def stream(value):
    fields(value, "schema phase token_dtype tokenizer_sha256 documents tokens targets index_root window_policy files", "stream")
    if value["schema"] != "ovl.stream.v1" or value["phase"] not in ("wikipedia", "conversation"):
        raise EvidenceError("unsupported stream schema/phase")
    if value["token_dtype"] != "uint16-le" or value["window_policy"] != "per-document-overlap-one-v1":
        raise EvidenceError("unsupported stream encoding/window policy")
    for name in ("documents", "tokens", "targets"):
        integer(value[name], 1, 2**53 - 1, name)
    for name in ("index_root", "tokenizer_sha256"):
        require_digest(value[name])
    if {e.get("path") for e in value["files"]} != {"tokens.u16", "mask.u8", "documents.jsonl"}:
        raise EvidenceError("unexpected stream inventory")


def control(value):
    fields(value, "phase global_step phase_step cursor transcript schedule accumulation scaler", "control")
    if value["phase"] not in ("wikipedia", "conversation") or value["schedule"] != "constant-lr-v1" or value["accumulation"] != "none" or value["scaler"] != "none":
        raise EvidenceError("unsupported control state")
    for name in ("global_step", "phase_step", "cursor"):
        integer(value[name], 0, 2**53 - 1, name)
    require_digest(value["transcript"])
