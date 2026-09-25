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
import torch.nn.functional as F


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
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))


    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model] -> [B, T, d_model]
        mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, T, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.cfg.layer_norm_eps)  # [B, T, d_model]
        return self.w * x_norm + self.b  # [B, T, d_model]
        raise NotImplementedError


class Embed(nn.Module):
    """W_E [d_vocab, d_model]. tokens [B, T] -> [B, T, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.randn(cfg.d_vocab, cfg.d_model) * cfg.init_std)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, T] -> [B, T, d_model]
        return self.W_E[tokens]  # [B, T, d_model]
        raise NotImplementedError


class PosEmbed(nn.Module):
    """W_pos [n_ctx, d_model]. tokens [B, T] -> [B, T, d_model] (depends only on T)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) * cfg.init_std)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        return self.W_pos[:T][None].expand(B, T, -1)   # [T, d] -> [1, T, d] -> [B, T, d]
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
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * cfg.init_std)
        self.W_K = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * cfg.init_std)
        self.W_V = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * cfg.init_std)
        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) * cfg.init_std)
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model] -> [B, T, d_model]
        T = x.size(1)
        Dh = self.cfg.d_head
        # Compute Q, K, V
        q = torch.einsum("btd,hde->bhte", x, self.W_Q) + self.b_Q[None, :, None]  # [B, H, T, d_head]
        k = torch.einsum("btd,hde->bhte", x, self.W_K) + self.b_K[None, :, None]  # [B, H, T, d_head]
        v = torch.einsum("btd,hde->bhte", x, self.W_V) + self.b_V[None, :, None]  # [B, H, T, d_head]
        # Compute attention scores
        scores = torch.einsum("bhte,bhse->bhts", q, k) / (Dh ** 0.5)  # [B, H, T, T]
        # Apply causal mask
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)  # [T, T]
        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))  # [B, H, T, T]
        # Compute attention pattern
        
        pattern = torch.softmax(scores, dim=-1)       # keeps the gradient
        self.pattern = pattern.detach()               # stored copy, for inspection only
        z = torch.einsum("bhts,bhse->bhte", pattern, v)

        out = torch.einsum("bhte,hed->btd", z, self.W_O) + self.b_O[None, None, :]  # [B, T, d_model]
        return out
        raise NotImplementedError


class MLP(nn.Module):
    """W_in [d_model, d_mlp], b_in [d_mlp], W_out [d_mlp, d_model], b_out [d_model].

    out = gelu(x @ W_in + b_in) @ W_out + b_out, with the tanh-approximate GELU
    (F.gelu(..., approximate="tanh")), which is what GPT-2 used.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) * cfg.init_std)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) * cfg.init_std)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model] -> [B, T, d_model]
        hidden = F.gelu(x @ self.W_in + self.b_in, approximate="tanh")  # [B, T, d_mlp]
        out = hidden @ self.W_out + self.b_out  # [B, T, d_model]
        return out
        raise NotImplementedError


class Block(nn.Module):
    """Attributes: ln1, attn, and (unless cfg.attn_only) ln2, mlp. Pre-LN residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        if not cfg.attn_only:
            self.ln2 = LayerNorm(cfg)
            self.mlp = MLP(cfg)

    def forward(self, resid: Tensor) -> Tensor:
        # resid: [B, T, d_model]
        resid = resid + self.attn(self.ln1(resid))     # attention reads a normalized copy, writes back additively
        if not self.cfg.attn_only:
            resid = resid + self.mlp(self.ln2(resid))  # same pattern for the MLP
        return resid


class GPT(nn.Module):
    """Attributes: embed, pos_embed, blocks (nn.ModuleList), ln_final,
    W_U [d_model, d_vocab] (normal(0, init_std)), b_U [d_vocab] (zeros).

    forward(tokens [B, T]) -> logits [B, T, d_vocab]. Assert T <= n_ctx.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.W_U = nn.Parameter(torch.randn(cfg.d_model, cfg.d_vocab) * cfg.init_std)
        self.b_U = nn.Parameter(torch.zeros(cfg.d_vocab))

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, T] -> logits: [B, T, d_vocab]
        B, T = tokens.shape
        assert T <= self.cfg.n_ctx, f"Input sequence length {T} exceeds model context length {self.cfg.n_ctx}."
        resid = self.embed(tokens) + self.pos_embed(tokens)  # [B, T, d_model]
        for block in self.blocks:
            resid = block(resid)  # [B, T, d_model]
        resid = self.ln_final(resid)  # [B, T, d_model]
        logits = resid @ self.W_U + self.b_U  # [B, T, d_vocab]
        return logits
        raise NotImplementedError


def count_params(cfg: Config) -> int:
    D, V, H, Dh, M = cfg.d_model, cfg.d_vocab, cfg.n_heads, cfg.d_head, cfg.d_mlp
    ln = 2 * D                                    # w + b
    attn = 4 * H * D * Dh + 3 * H * Dh + D        # W_Q,W_K,W_V,W_O + b_Q,b_K,b_V + b_O
    mlp = D * M + M + M * D + D                   # W_in, b_in, W_out, b_out
    per_block = ln + attn + (0 if cfg.attn_only else ln + mlp)   # ln1+attn, then ln2+mlp
    embed = V * D + cfg.n_ctx * D                 # W_E + W_pos
    unembed = ln + D * V + V                      # ln_final + W_U + b_U
    return embed + cfg.n_layers * per_block + unembed
