"""Exercise 2 — backprop by hand.

You will not hand-write backward passes at work. You do this once so that
"the gradient is (softmax - onehot) / N" and "dW = grad_out.T @ x" are facts
you can derive on a whiteboard, and so a gradient-check is a tool you reach for
when a custom op misbehaves.

Run:  uv run pytest tests/test_part1.py -k ex02 -q
"""

from typing import Callable

import torch
from torch import Tensor


def softmax_cross_entropy_grad(logits: Tensor, targets: Tensor) -> Tensor:
    """dL/dlogits for L = F.cross_entropy(logits, targets) (mean reduction).

    logits: [N, C], targets: [N] long  ->  [N, C]
    No autograd. Write the formula.
    """
    raise NotImplementedError


def linear_backward(x: Tensor, W: Tensor, b: Tensor, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Backward for y = x @ W.T + b  (the nn.Linear convention: W is [out, in]).

    x: [N, in], W: [out, in], b: [out], grad_out: [N, out]
    -> (grad_x [N, in], grad_W [out, in], grad_b [out])
    """
    raise NotImplementedError


def numerical_grad(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-6) -> Tensor:
    """Central-difference gradient of scalar f at x (same shape as x).

    Use float64. This is O(numel) forward passes: fine for tests, never for training.
    """
    raise NotImplementedError
