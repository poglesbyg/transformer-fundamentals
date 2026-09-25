import math

import pytest
import torch
import torch.nn.functional as F

from part2_gpt.data import get_batch
from part2_gpt.model import GPT, Config
from part2_gpt.sample import sample
from part2_gpt.train import (
    MetricsLogger,
    TrainConfig,
    configure_optimizer,
    global_grad_norm,
    lm_loss,
    lr_at,
    train,
    update_ratio,
)

CFG = Config(d_vocab=16, d_model=32, n_layers=2, n_heads=4, d_mlp=64, n_ctx=16)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------- data

def test_get_batch_shift_and_bounds():
    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    starts = set()
    for _ in range(300):
        x, y = get_batch(data, block_size=8, batch_size=16, generator=g)
        assert x.shape == y.shape == (16, 8)
        assert x.dtype == torch.long
        assert torch.equal(y, x + 1), "y must be x shifted by one position"
        assert (y <= 99).all()
        starts.update(x[:, 0].tolist())
    assert starts == set(range(92)), "every valid window start (0..len-block_size-1) should be reachable"


def test_get_batch_uses_generator():
    data = torch.arange(1000)
    a = get_batch(data, 8, 4, torch.Generator().manual_seed(1))
    b = get_batch(data, 8, 4, torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0])


# ---------------------------------------------------------------- pieces

def test_lm_loss():
    logits, targets = torch.randn(3, 5, 7), torch.randint(0, 7, (3, 5))
    torch.testing.assert_close(lm_loss(logits, targets), F.cross_entropy(logits.permute(0, 2, 1), targets))


def test_configure_optimizer_groups():
    m = GPT(CFG)
    opt = configure_optimizer(m, lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    ids = {id(p): p for p in m.parameters()}
    seen = set()
    for group in opt.param_groups:
        for p in group["params"]:
            seen.add(id(p))
            expected = 0.1 if p.ndim >= 2 else 0.0
            assert group["weight_decay"] == expected, f"param of shape {tuple(p.shape)} has wd {group['weight_decay']}"
    assert seen == set(ids), "every parameter must be in exactly one group"
    assert opt.defaults["betas"] == (0.9, 0.95)


def test_lr_schedule():
    kw = dict(max_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)
    assert lr_at(0, **kw) == pytest.approx(0.0)
    assert lr_at(5, **kw) == pytest.approx(0.5)
    assert lr_at(10, **kw) == pytest.approx(1.0)
    assert lr_at(60, **kw) == pytest.approx(0.55)  # halfway through cosine
    assert lr_at(110, **kw) == pytest.approx(0.1)
    assert lr_at(500, **kw) == pytest.approx(0.1)
    lrs = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "must decay monotonically after warmup"


def test_global_grad_norm_matches_clip():
    m = GPT(CFG)
    toks = torch.randint(0, CFG.d_vocab, (4, 16))
    lm_loss(m(toks[:, :-1]), toks[:, 1:]).backward()
    expected = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=float("inf")).item()
    assert global_grad_norm(m) == pytest.approx(expected, rel=1e-5)


def test_update_ratio():
    m = torch.nn.Linear(3, 3)
    before = [p.detach().clone() for p in m.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(1.1)
    assert update_ratio(before, m) == pytest.approx(0.1, rel=1e-5)


# ---------------------------------------------------------------- the loop

def test_train_learns_a_pattern():
    """A period-5 sequence is perfectly predictable after the first few tokens.
    A working loop drives the loss well below ln(16) in a few hundred steps."""
    data = torch.tensor([3, 1, 4, 1, 5] * 400)
    model = GPT(CFG)
    tcfg = TrainConfig(steps=200, batch_size=16, max_lr=3e-3, min_lr=3e-4, warmup_steps=20, eval_interval=50, eval_batches=4, log_interval=10)
    logger = MetricsLogger(None)
    train(model, data, data, tcfg, logger)

    steps = [r["step"] for r in logger.rows]
    assert steps and steps[0] == 0 and max(steps) >= 190
    for k in ["lr", "train_loss", "grad_norm", "update_ratio"]:
        assert all(k in r for r in logger.rows), f"every log row needs {k}"
    vals = [r["val_loss"] for r in logger.rows if r.get("val_loss") is not None]
    assert len(vals) >= 4
    assert abs(logger.rows[0]["train_loss"] - math.log(16)) < 0.3
    assert vals[-1] < 0.2
    assert model.training, "leave the model in train mode"


# ---------------------------------------------------------------- sampling

def test_sample_greedy_and_topk1_agree():
    m = GPT(CFG)
    prompt = torch.tensor([1, 2, 3])
    greedy = sample(m, prompt, 20, temperature=0.0)
    top1 = sample(m, prompt, 20, temperature=1.0, top_k=1, generator=torch.Generator().manual_seed(0))
    assert greedy.shape == (23,)
    assert torch.equal(greedy[:3], prompt)
    assert torch.equal(greedy, top1)


def test_sample_crops_context_and_is_seeded():
    m = GPT(CFG)
    prompt = torch.randint(0, CFG.d_vocab, (CFG.n_ctx,))
    a = sample(m, prompt, 40, generator=torch.Generator().manual_seed(3))  # would crash without cropping
    b = sample(m, prompt, 40, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
