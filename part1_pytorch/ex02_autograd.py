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
    # Compute the softmax of the logits
    softmax = torch.softmax(logits, dim=-1)
    # Create a one-hot encoding of the targets
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(-1, targets.unsqueeze(-1), 1.0)
    # Compute the gradient
    grad = (softmax - one_hot) / logits.size(0)
    return grad
    raise NotImplementedError


def linear_backward(x: Tensor, W: Tensor, b: Tensor, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Backward for y = x @ W.T + b  (the nn.Linear convention: W is [out, in]).

    x: [N, in], W: [out, in], b: [out], grad_out: [N, out]
    -> (grad_x [N, in], grad_W [out, in], grad_b [out])
    """
    # Compute gradients
    grad_x = grad_out @ W  # [N, in]
    grad_W = grad_out.t() @ x  # [out, in]
    grad_b = grad_out.sum(dim=0)  # [out]
    return grad_x, grad_W, grad_b
    raise NotImplementedError


def numerical_grad(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-6) -> Tensor:
    """Central-difference gradient of scalar f at x (same shape as x).

    Use float64. This is O(numel) forward passes: fine for tests, never for training.
    """
    x = x.double()  # Ensure float64 for numerical stability
    grad = torch.zeros_like(x, dtype=torch.float64)
    # Iterate over all indices in x
    for idx in range(x.numel()):
        # Create a copy of x to perturb
        x_plus = x.clone()
        x_minus = x.clone()
        # Perturb the current index by eps
        x_plus.view(-1)[idx] += eps
        x_minus.view(-1)[idx] -= eps
        # Compute the function values at the perturbed points
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        # Compute the central difference
        grad.view(-1)[idx] = (f_plus - f_minus) / (2 * eps)
    return grad
    raise NotImplementedError
