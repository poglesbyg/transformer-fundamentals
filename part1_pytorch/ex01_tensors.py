"""Exercise 1 — tensor fluency.

Every function here must be written WITHOUT Python loops over tensor elements.
Reach for broadcasting, indexing, `einsum`, `gather`, `torch.tril`, etc.

Run:  pytest tests/test_part1.py -k ex01 -q

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
    raise NotImplementedError


def pairwise_sq_dists(x: Tensor, y: Tensor) -> Tensor:
    """x: [n, d], y: [m, d]  ->  [n, m] squared Euclidean distances.

    Do it two ways in your head: via broadcasting (x[:, None] - y[None]) and via
    the expansion |x|^2 - 2 x·y + |y|^2. Implement either; know the memory
    difference between them.
    """
    raise NotImplementedError


def gather_logprobs(logits: Tensor, targets: Tensor) -> Tensor:
    """logits: [B, T, V] float, targets: [B, T] long  ->  [B, T] log p(target).

    The negative mean of this is exactly F.cross_entropy. Don't call F.cross_entropy.
    """
    raise NotImplementedError


def causal_mask(T: int) -> Tensor:
    """-> bool [T, T], True where query position q must NOT attend to key k (k > q)."""
    raise NotImplementedError


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax that doesn't overflow for large inputs and handles -inf entries.

    Must match torch.softmax on e.g. torch.tensor([1000., 1001., -float('inf')]).
    Don't call torch.softmax / F.softmax.
    """
    raise NotImplementedError
