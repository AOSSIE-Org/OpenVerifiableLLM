import hashlib

from openverifiablellm.model import ModelConfig, VerifiableLLM


def get_model_hash(model):
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def test_model_initialization_is_deterministic():
    config = ModelConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
    )

    model1 = VerifiableLLM(config)
    model1.init_weights_deterministically(seed=42)
    hash1 = get_model_hash(model1)

    model2 = VerifiableLLM(config)
    model2.init_weights_deterministically(seed=42)
    hash2 = get_model_hash(model2)

    model3 = VerifiableLLM(config)
    model3.init_weights_deterministically(seed=99)
    hash3 = get_model_hash(model3)

    assert hash1 == hash2
    assert hash1 != hash3

    # Re-initialization regression: mutate model1 and verify reinit restores original state
    import torch

    for param in model1.parameters():
        param.data.add_(torch.randn_like(param.data))
    hash_mutated = get_model_hash(model1)
    assert hash_mutated != hash1, "Mutation should change the hash"

    model1.init_weights_deterministically(seed=42)
    hash_reinit = get_model_hash(model1)
    assert hash_reinit == hash1, "Reinit should restore original deterministic state"
