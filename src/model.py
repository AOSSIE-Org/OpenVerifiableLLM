# Model zoo for OpenVerifiableLLM.
#
# The transformer (TinyGPT) is derived from Andrej Karpathy's nanoGPT
# (https://github.com/karpathy/nanoGPT) but is extended here, not vendored:
# num_layers is configurable so the same class spans gpt10m (~11M params) and
# gpt50m (~57M). Alongside it sit three other op-families on purpose, because
# each behaves differently under hardware nondeterminism:
#
#   * MLPLanguageModel  - position-wise, no cross-token mixing. The control:
#                         Embedding + Linear + GELU only. Under fp32 +
#                         deterministic algorithms it is reliably bitwise.
#   * TinyGPT           - attention (q@k, softmax) + embedding-gather atomics.
#   * LSTMLanguageModel - cuDNN recurrent kernels (the classic "RNN is not
#                         deterministic on GPU" caveat).
#   * TinyCNN           - cuDNN convolutions (a different nondeterminism
#                         mechanism again; CIFAR-10 modality).
#
# build_model(name, vocab_size, cfg) is the single construction site used by the
# runner, the sweep and the audit.

import math
import typing

import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Transformer (nanoGPT-derived, num_layers configurable)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    # Type hint for the dynamically registered buffer
    bias: torch.Tensor

    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = embed_dim

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    """nanoGPT-style decoder. num_layers is configurable (this is the change that
    turns the old single-block toy into a multi-megabyte, multi-chunk artifact)."""

    def __init__(
        self,
        vocab_size,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        max_seq_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, embed_dim),
                wpe=nn.Embedding(max_seq_len, embed_dim),
                h=nn.ModuleList(
                    [
                        Block(embed_dim, num_heads, max_seq_len, dropout)
                        for _ in range(num_layers)
                    ]
                ),
                ln_f=nn.LayerNorm(embed_dim),
            )
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        wte = typing.cast(nn.Embedding, self.transformer["wte"])
        wpe = typing.cast(nn.Embedding, self.transformer["wpe"])
        h = typing.cast(nn.ModuleList, self.transformer["h"])
        ln_f = typing.cast(nn.LayerNorm, self.transformer["ln_f"])

        x = wte(idx) + wpe(pos)
        for block in h:
            x = block(x)
        x = ln_f(x)
        return self.lm_head(x)


# --------------------------------------------------------------------------- #
# MLP language model (the op-family control)
# --------------------------------------------------------------------------- #
class MLPLanguageModel(nn.Module):
    """Position-wise MLP LM. Each position is processed independently (no
    attention, no recurrence, no convolution), so the only floating-point
    reductions are dense matmuls. This is the reference against which the special
    ops of the other models are compared."""

    def __init__(
        self,
        vocab_size,
        embed_dim=384,
        hidden_dim=1024,
        num_layers=2,
        max_seq_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)
        layers = []
        d = embed_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.mlp(x)
        return self.lm_head(x)


# --------------------------------------------------------------------------- #
# LSTM language model (cuDNN recurrent caveats)
# --------------------------------------------------------------------------- #
class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=384,
        hidden_dim=512,
        num_layers=2,
        max_seq_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx):
        x = self.wte(idx)
        out, _ = self.lstm(x)
        return self.lm_head(out)


# --------------------------------------------------------------------------- #
# CNN classifier (CIFAR-10; cuDNN conv nondeterminism) - stretch modality
# --------------------------------------------------------------------------- #
class TinyCNN(nn.Module):
    def __init__(self, channels=64, num_classes=10, **_ignore):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 32 -> 16
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * c * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# Factory + helpers
# --------------------------------------------------------------------------- #
_GPT_ALIASES = {"gpt", "gpt10m", "gpt50m", "gpt120m", "tinygpt"}


def build_model(name, vocab_size, cfg):
    """Single construction site for every model in the matrix.

    ``cfg`` is a dict from config.model_config(name): it already carries the
    architecture preset merged over the canonical config.
    """
    name = name.lower()
    if name in _GPT_ALIASES:
        return TinyGPT(
            vocab_size,
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
        )
    if name == "mlp":
        return MLPLanguageModel(
            vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
        )
    if name == "lstm":
        return LSTMLanguageModel(
            vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg.get("hidden_dim", 512),
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
        )
    if name == "cnn":
        return TinyCNN(
            channels=cfg.get("channels", 64),
            num_classes=cfg.get("num_classes", 10),
        )
    raise ValueError(f"unknown model {name!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


IS_VISION = {"cnn"}


def is_vision_model(name):
    """True for image models (different batch shape / dataset / loss path)."""
    return name.lower() in IS_VISION
