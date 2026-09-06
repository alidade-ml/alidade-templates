"""Transformer building blocks.

Harvested from a real research codebase and cut down until it trains on a CPU in
seconds. The shapes and the maths are the originals; what was removed is the
FlashAttention path, which needs a GPU and a compiled kernel, and the
HuggingFace `PreTrainedModel` scaffolding that came with it.

That removal is why this runs anywhere. The eager attention below is the same
path the original used whenever flash was unavailable.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalise by root-mean-square, with no mean subtraction and no bias.

    Cheaper than LayerNorm and, in practice, no worse. The float32 cast is not
    optional: the variance of a half-precision activation underflows.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings, precomputed to `max_positions`.

    Position enters through a rotation of the query and key vectors rather than
    an added embedding, so relative distance falls out of the dot product.
    """

    def __init__(self, dim: int, max_positions: int = 512, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_positions, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype):
        return (
            self.cos_cached[:seq_len].to(dtype),
            self.sin_cached[:seq_len].to(dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class GatedMLP(nn.Module):
    """Two projections multiplied together, one of them activated.

    The gate lets the block suppress a channel outright, which a single
    activated projection cannot do.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Multi-head self-attention, eager.

    The original dispatched to FlashAttention when a GPU and the kernel were
    both present and fell back to this otherwise. Only the fallback is kept, so
    the model runs on a laptop.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_positions: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_positions)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        b, s, _ = x.shape
        shape = (b, s, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)

        cos, sin = self.rotary(s, x.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # Additive, not multiplicative: a masked position must reach -inf
            # before the softmax, or padding still contributes probability.
            scores = scores + attention_mask[:, None, None, :]
        weights = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        out = (weights @ v).transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)
