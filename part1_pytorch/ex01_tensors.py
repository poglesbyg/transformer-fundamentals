"""Exercise 1 — tensor fluency.

Every function here must be written WITHOUT Python loops over tensor elements.
Reach for broadcasting, indexing, `einsum`, `gather`, `torch.tril`, etc.

Run:  uv run pytest tests/test_part1.py -k ex01 -q

Why these five: each one shows up, almost verbatim, inside a transformer.
  batched_outer      -> the q·k score computation, one step removed
  pairwise_sq_dists  -> broadcasting discipline; the classic shape bug factory
  gather_logprobs    -> how cross-entropy actually picks out the target logit
  causal_mask        -> what "decoder-only" means in one line
  stable_softmax     -> why softmax never overflows in practice (and -inf masks)
"""

import torch
from torch import Tensor


def batched_outer(a: Tensor, b: Tensor) -> Tensor:
    """a: [B, n], b: [B, m]  ->  [B, n, m] with out[i, j, k] = a[i, j] * b[i, k]."""
    return a.unsqueeze(-1) * b.unsqueeze(-2)
    raise NotImplementedError


def pairwise_sq_dists(x: Tensor, y: Tensor) -> Tensor:
    """x: [n, d], y: [m, d]  ->  [n, m] squared Euclidean distances.

    Do it two ways in your head: via broadcasting (x[:, None] - y[None]) and via
    the expansion |x|^2 - 2 x·y + |y|^2. Implement either; know the memory
    difference between them.
    """
    x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # [n, 1]
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)  # [1, m]
    cross_term = x @ y.t()  # [n, m]
    return x_norm - 2 * cross_term + y_norm
    raise NotImplementedError


def gather_logprobs(logits: Tensor, targets: Tensor) -> Tensor:
    """logits: [B, T, V] float, targets: [B, T] long  ->  [B, T] log p(target).

    The negative mean of this is exactly F.cross_entropy. Don't call F.cross_entropy.
    """
    # Use gather to select the logit corresponding to the target class for each position
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    gathered_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    return gathered_log_probs
    raise NotImplementedError


def causal_mask(T: int) -> Tensor:
    """-> bool [T, T], True where query position q must NOT attend to key k (k > q)."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    raise NotImplementedError


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax that doesn't overflow for large inputs and handles -inf entries.

    Must match torch.softmax on e.g. torch.tensor([1000., 1001., -float('inf')]).
    Don't call torch.softmax / F.softmax.
    """
    # Subtract the max for numerical stability
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - max_x
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    softmax = exp_x / sum_exp_x
    return softmax
    raise NotImplementedError
