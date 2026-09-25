"""Debugging drill 06.

Run with:
    uv run python -m part3_debugging.drill_06
    uv run python -m part2_gpt.plot runs/drill_00_baseline runs/drill_06

Every drill uses your GPT from part2_gpt/model.py and writes the same metrics.
Read DRILLS.md for how to work through them.
"""

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from part2_gpt.data import load_shakespeare
from part2_gpt.model import GPT, Config
from part2_gpt.train import MetricsLogger

STEPS = 600
BATCH = 32
LR = 3e-3
WARMUP = 50


def batch(data, T, B, g):
    ix = torch.randint(0, len(data) - T, (B,), generator=g).tolist()
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + T] for i in ix])
    return x, y


def init_weights(model):
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.02)


def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return 0.1 * LR + 0.9 * LR * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))


@torch.no_grad()
def evaluate(model, data, T, g):
    model.eval()
    out = sum(loss_fn(model(x), y).item() for x, y in (batch(data, T, BATCH, g) for _ in range(10))) / 10
    model.train()
    return out


def main():
    torch.manual_seed(0)
    g, g_eval = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    tok, train, val = load_shakespeare()
    cfg = Config(d_vocab=tok.d_vocab, d_model=64, n_layers=2, n_heads=4, d_mlp=256, n_ctx=64)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    torch.manual_seed(0)  # re-seed so every drill starts from identical weights
    model = GPT(cfg)
    init_weights(model)
    logger = MetricsLogger(Path("runs") / Path(sys.argv[0]).stem)
    t0 = time.time()

    for step in range(STEPS):
        lr = lr_at(step)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = batch(train, cfg.n_ctx, BATCH, g)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        ratio = math.sqrt(
            sum((p.detach() - b).pow(2).sum().item() for p, b in zip(model.parameters(), before))
            / sum(b.pow(2).sum().item() for b in before)
        )
        row = dict(step=step, lr=lr, train_loss=loss.item(), grad_norm=grad_norm, update_ratio=ratio)
        if step % 50 == 0 or step == STEPS - 1:
            row["val_loss"] = evaluate(model, val, cfg.n_ctx, g_eval)
            print(f"step {step:4d}  train {loss.item():.3f}  val {row['val_loss']:.3f}  |g| {grad_norm:.3f}  upd {ratio:.1e}")
        logger.log(**row)

    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
