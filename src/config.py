import json
import hashlib
import os

# The canonical ("immutable") training configuration for the segmented-replay
# audit pipeline: the gpt10m-on-shakespeare reference run. run_experiment.py and
# sweep.py override these per matrix cell; reproducibility.py / global_manifest.py
# use them as-is. Dims are chosen so the safetensors checkpoint is large enough
# for a NON-degenerate Merkle tree (~43 MB -> ~43 one-MB chunks), unlike the old
# 21 KB toy that collapsed the tree to a single chunk.
TRAIN_CONFIG = {
    "embed_dim": 384,
    "num_heads": 6,
    "num_layers": 6,
    "max_seq_len": 256,
    "block_size": 128,
    "batch_size": 16,
    "dropout": 0.1,
    "lr": 1e-3,            # 1e-3, NOT 1e-2: 1e-2 NaNs a ~10M-param transformer
    "optimizer": "Adam",
    "seed": 99,            # never 42
    "total_steps": 200,
    "checkpoint_step": 100,
}

# Per-model architecture presets. mlp / gpt* / lstm share the char-level text
# pipeline; cnn is a separate modality (CIFAR-10). The spread of op-families is
# deliberate: each has different determinism behaviour, which is the debate.
MODEL_PRESETS = {
    "mlp":    {"embed_dim": 384, "hidden_dim": 1024, "num_layers": 2},  # control: no cross-token mixing
    "gpt10m": {"embed_dim": 384, "num_heads": 6,  "num_layers": 6},     # attention + embedding atomics (~11M)
    "gpt50m": {"embed_dim": 768, "num_heads": 12, "num_layers": 8},     # scale axis (~57M)
    "lstm":   {"embed_dim": 384, "hidden_dim": 512, "num_layers": 2},   # cuDNN recurrent caveats
    "cnn":    {"channels": 64, "num_classes": 10},                      # cuDNN conv nondeterminism
}


def _coerce(value):
    """Coerce an env-var string to int/float when it looks numeric."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def effective_config(**overrides):
    """TRAIN_CONFIG with OVL_* env-var and explicit overrides applied.

    Lets a CPU smoke run shrink the workload (e.g. ``OVL_TOTAL_STEPS=8
    OVL_BATCH_SIZE=4``) without editing the committed canonical config.
    Precedence: explicit overrides > env vars > TRAIN_CONFIG.
    """
    cfg = dict(TRAIN_CONFIG)
    for key in list(cfg):
        env_key = "OVL_" + key.upper()
        if env_key in os.environ:
            cfg[key] = _coerce(os.environ[env_key])
    cfg.update(overrides)
    return cfg


def model_config(model_name, **overrides):
    """Effective config merged with a model's architecture preset.

    Explicit overrides win over the preset; the preset (architecture) wins over
    the env/canonical defaults for the keys it defines.
    """
    cfg = effective_config(**overrides)
    cfg.update(MODEL_PRESETS.get(model_name, {}))
    cfg.update(overrides)
    cfg["model"] = model_name
    return cfg


def get_config_hash():
    """Returns a deterministic SHA-256 hash of the canonical configuration dict."""
    encoded = json.dumps(TRAIN_CONFIG, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


# scaled-baseline config (gpt10m default); see model_config() for per-model dims
