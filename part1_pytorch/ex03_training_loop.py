"""Exercise 3 — the training loop, with nothing hidden.

Write the five lines every PyTorch training loop is made of, on a problem small
enough that you can watch it work: a 2-D, 3-arm spiral that a linear model
cannot solve and a small MLP can.

Run:  uv run pytest tests/test_part1.py -k ex03 -q
Then: uv run python -m part1_pytorch.ex03_training_loop   (prints loss + accuracy)
"""

import math

import torch
from torch import Tensor, nn


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def make_spirals(n_per_class: int = 300, n_classes: int = 3, noise: float = 0.15, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = torch.linspace(0.05, 1.0, n_per_class)
        theta = torch.linspace(c * 2 * math.pi / n_classes, c * 2 * math.pi / n_classes + 4.0, n_per_class)
        theta = theta + noise * torch.randn(n_per_class, generator=g)
        xs.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1))
        ys.append(torch.full((n_per_class,), c))
    return torch.cat(xs), torch.cat(ys)


def make_mlp(d_in: int = 2, d_hidden: int = 64, n_classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, n_classes))


def train_classifier(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    *,
    steps: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 128,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Train `model` with Adam + cross-entropy on random minibatches; return per-step losses.

    Sample each minibatch with torch.randint(..., generator=generator).
    Be able to say, for each line you write, what breaks if you delete it.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        # Sample a random minibatch
        indices = torch.randint(0, X.size(0), (batch_size,), generator=generator)
        x_batch = X[indices]
        y_batch = y[indices]
        # Forward pass
        logits = model(x_batch)
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def accuracy(model: nn.Module, X: Tensor, y: Tensor) -> float:
    """Compute the fraction of correct predictions on (X, y)."""
    logits = model(X)
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == y).sum().item()
    return correct / y.size(0)
    raise NotImplementedError


if __name__ == "__main__":
    g = set_seed(0)
    X, y = make_spirals()
    model = make_mlp()
    losses = train_classifier(model, X, y, generator=g)
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}   (chance = ln 3 = {math.log(3):.3f})")
    print(f"train accuracy {accuracy(model, X, y):.3f}")
