"""A small masked-language model.

Encoder-only, bidirectional, trained by hiding tokens and predicting them. Small
enough to train on a CPU, which is the point: the template has to run in CI, or
it drifts out of date without anyone noticing.

Change `TinyBertConfig` to make it bigger. Change `train_bert.py` to feed it
your own data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torch_models.layers import RMSNorm, Attention, GatedMLP


@dataclass
class TinyBertConfig:
    vocab_size: int = 1024
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_positions: int = 128
    mask_token_id: int = 3


class EncoderBlock(nn.Module):
    """Pre-norm: normalise going in, add the residual raw.

    Post-norm normalises the sum instead, which needs a learning-rate warmup to
    train at depth. Pre-norm is the reason this converges without one.
    """

    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg.hidden_size, cfg.num_heads, cfg.max_positions)
        self.mlp_norm = RMSNorm(cfg.hidden_size)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        return x + self.mlp(self.mlp_norm(x))


class TinyBertForMaskedLM(nn.Module):
    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Tied to the input embedding: the same matrix maps token to vector and
        # vector back to token, which halves the parameters and trains better at
        # this size.
        self.head.weight = self.embeddings.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small normal init, std 0.02.

        Not cosmetic. torch defaults nn.Embedding to N(0, 1), and because the
        head is tied to the embedding the logit for token t contains
        ||E[t]||^2, which at hidden_size=128 is around 128. That single term
        swamps the softmax, the model reproduces its input, and the loss starts
        near zero instead of ln(vocab_size).
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask.to(self.embeddings.weight.dtype)) * -1e9

        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.norm(x))

        loss = None
        if labels is not None:
            # -100 is torch's ignore index. Only masked positions carry a label,
            # so the loss is computed over those alone.
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), labels.view(-1)
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
