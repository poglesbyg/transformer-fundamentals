"""A GPT-2-style decoder-only transformer, written by you.

Weight shapes follow the TransformerLens convention that ARENA uses
(per-head W_Q [n_heads, d_model, d_head], W_O [n_heads, d_head, d_model], ...),
so when you open ARENA the notation is already yours.

Architecture (pre-LN, as in GPT-2):

    tokens [B, T]
      -> embed (W_E) + positional embed (W_pos)            resid [B, T, d_model]
      -> n_layers x Block:
             resid = resid + Attention(LN1(resid))
             resid = resid + MLP(LN2(resid))               (skip MLP if cfg.attn_only)
      -> LN_final -> unembed (W_U, b_U)                    logits [B, T, d_vocab]

Run:  uv run pytest tests/test_model.py -q

Rules for yourself:
  * Write the shape of every intermediate in a comment the first time through.
  * Don't use nn.MultiheadAttention, nn.LayerNorm, nn.Linear or
    F.scaled_dot_product_attention in this file. The tests use them as the reference.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class Config:
    d_vocab: int = 65
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 128
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    attn_only: bool = False  # Part 4 trains an attention-only model

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class LayerNorm(nn.Module):
    """Parameters: w [d_model] (init ones), b [d_model] (init zeros).

    Normalize over the last dim with the *biased* variance, then scale and shift.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO: self.w, self.b as nn.Parameter

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model] -> [B, T, d_model]
        raise NotImplementedError


class Embed(nn.Module):
    """W_E [d_vocab, d_model]. tokens [B, T] -> [B, T, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO: self.W_E, normal(0, cfg.init_std)

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError


class PosEmbed(nn.Module):
    """W_pos [n_ctx, d_model]. tokens [B, T] -> [B, T, d_model] (depends only on T)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO: self.W_pos, normal(0, cfg.init_std)

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError


class Attention(nn.Module):
    """Causal multi-head self-attention.

    Parameters (names and shapes are checked by the tests):
        W_Q, W_K, W_V : [n_heads, d_model, d_head]   init normal(0, init_std)
        b_Q, b_K, b_V : [n_heads, d_head]            init zeros
        W_O           : [n_heads, d_head, d_model]   init normal(0, init_std)
        b_O           : [d_model]                    init zeros

    Contract:
        forward(x [B, T, d_model]) -> [B, T, d_model]
        After every forward, self.pattern holds the post-softmax attention
        probabilities, detached, shape [B, n_heads, T_q, T_k]. Part 4 reads it.

    Steps you must be able to recite:
        q, k, v   = x @ W_{Q,K,V} + b             [B, T, H, d_head]
        scores    = q · k / sqrt(d_head)           [B, H, T_q, T_k]
        mask      = scores where k > q  ->  -inf    (before softmax; why not after?)
        pattern   = softmax over T_k
        z         = pattern @ v                    [B, T_q, H, d_head]
        out       = sum_h z_h @ W_O[h] + b_O       [B, T, d_model]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pattern: Tensor | None = None
        # TODO: parameters

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MLP(nn.Module):
    """W_in [d_model, d_mlp], b_in [d_mlp], W_out [d_mlp, d_model], b_out [d_model].

    out = gelu(x @ W_in + b_in) @ W_out + b_out, with the tanh-approximate GELU
    (F.gelu(..., approximate="tanh")), which is what GPT-2 used.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Block(nn.Module):
    """Attributes: ln1, attn, and (unless cfg.attn_only) ln2, mlp. Pre-LN residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO

    def forward(self, resid: Tensor) -> Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    """Attributes: embed, pos_embed, blocks (nn.ModuleList), ln_final,
    W_U [d_model, d_vocab] (normal(0, init_std)), b_U [d_vocab] (zeros).

    forward(tokens [B, T]) -> logits [B, T, d_vocab]. Assert T <= n_ctx.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # TODO

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError


def count_params(cfg: Config) -> int:
    """Closed-form parameter count for GPT(cfg), from the shapes above — not by
    instantiating the model. Interviewers ask for this for GPT-2 small (~124M);
    be able to do it on paper and say where most of the parameters live.
    """
    raise NotImplementedError
