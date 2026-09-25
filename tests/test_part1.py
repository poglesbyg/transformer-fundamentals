import math

import pytest
import torch
import torch.nn.functional as F

from part1_pytorch import ex01_tensors as ex01
from part1_pytorch import ex02_autograd as ex02
from part1_pytorch import ex03_training_loop as ex03


# ---------------------------------------------------------------- ex01

def test_ex01_batched_outer():
    a, b = torch.randn(4, 3), torch.randn(4, 5)
    out = ex01.batched_outer(a, b)
    assert out.shape == (4, 3, 5)
    torch.testing.assert_close(out, torch.stack([torch.outer(a[i], b[i]) for i in range(4)]))


def test_ex01_pairwise_sq_dists():
    x, y = torch.randn(7, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    torch.testing.assert_close(ex01.pairwise_sq_dists(x, y), torch.cdist(x, y) ** 2)


def test_ex01_gather_logprobs():
    logits, targets = torch.randn(2, 6, 11), torch.randint(0, 11, (2, 6))
    lp = ex01.gather_logprobs(logits, targets)
    assert lp.shape == (2, 6)
    torch.testing.assert_close(-lp.mean(), F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1)))


def test_ex01_causal_mask():
    m = ex01.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]


@pytest.mark.parametrize("x", [torch.randn(3, 9), torch.tensor([[1000.0, 1001.0, -float("inf")]]), torch.tensor([[-1000.0, -1000.5, -999.0]])])
def test_ex01_stable_softmax(x):
    out = ex01.stable_softmax(x, dim=-1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


# ---------------------------------------------------------------- ex02

def test_ex02_softmax_ce_grad():
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    F.cross_entropy(logits, targets).backward()
    torch.testing.assert_close(ex02.softmax_cross_entropy_grad(logits.detach(), targets), logits.grad)


def test_ex02_linear_backward():
    x = torch.randn(6, 4, requires_grad=True)
    lin = torch.nn.Linear(4, 3)
    grad_out = torch.randn(6, 3)
    lin(x).backward(grad_out)
    gx, gW, gb = ex02.linear_backward(x.detach(), lin.weight.detach(), lin.bias.detach(), grad_out)
    torch.testing.assert_close(gx, x.grad)
    torch.testing.assert_close(gW, lin.weight.grad)
    torch.testing.assert_close(gb, lin.bias.grad)


def test_ex02_numerical_grad():
    x = torch.randn(3, 4, dtype=torch.float64)

    def f(t):
        return (t.sin() * t).sum() + (t ** 2).mean()

    x_ = x.clone().requires_grad_(True)
    f(x_).backward()
    torch.testing.assert_close(ex02.numerical_grad(f, x), x_.grad, atol=1e-6, rtol=1e-5)
    assert torch.equal(x, x.clone()), "numerical_grad must not leave x modified"


# ---------------------------------------------------------------- ex03

def test_ex03_training_solves_spirals():
    g = ex03.set_seed(0)
    X, y = ex03.make_spirals()
    model = ex03.make_mlp()
    losses = ex03.train_classifier(model, X, y, steps=1500, generator=g)
    assert len(losses) == 1500
    assert abs(losses[0] - math.log(3)) < 0.15, "step-0 loss of a fresh classifier should be near ln(n_classes)"
    assert sum(losses[-50:]) / 50 < 0.2
    assert ex03.accuracy(model, X, y) > 0.95
