from typing import Optional

import torch
import torch.nn as nn

from .attention import Attention, precompute_freqs_cis
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class VerifiableLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        if seqlen > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seqlen} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )
        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )

        return logits, loss

    def init_weights_deterministically(self, seed: int):
        """
        Initializes weights using a specific random seed.
        This guarantees the model starts completely untrained from an exact, reproducible baseline state,
        which is a core requirement of the Open-Everything Verifiable LLM protocol.
        """
        # Save original state of RNGs
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            # Set strict seed for reproducibility of initialization
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(
                        module.weight, mean=0.0, std=self.config.initializer_range
                    )
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(
                        module.weight, mean=0.0, std=self.config.initializer_range
                    )
                elif isinstance(module, RMSNorm):
                    torch.nn.init.ones_(module.weight)
        finally:
            # Restore RNG to avoid changing outer application state unintentionally
            torch.set_rng_state(cpu_rng_state)
            if gpu_rng_states is not None:
                torch.cuda.set_rng_state_all(gpu_rng_states)
