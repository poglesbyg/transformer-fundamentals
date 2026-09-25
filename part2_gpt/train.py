"""Train your GPT on Tiny Shakespeare, and log the numbers you'd need to debug it.

Run:  uv run pytest tests/test_train.py -q
Then: uv run python -m part2_gpt.train --name baseline
      uv run python -m part2_gpt.plot runs/baseline

The default config (~0.8M params) takes ~7 minutes on a 4-core CPU and reaches
val loss ~1.5, with train loss ~1.25. Samples look like Shakespeare: real
character names and mostly real words, with no sense. Explain the train/val gap
and name two things that would shrink it.

Before you run it, write down what loss you expect at step 0, and what loss a
model that only learned character frequencies would get. REFERENCE_LOSSES
computes both. If step 0 is far from ln(d_vocab), stop and find out why.
"""

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from part2_gpt.data import get_batch, load_shakespeare
from part2_gpt.model import GPT, Config


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    seed: int = 0


# --------------------------------------------------------------------------- #
# Pieces you implement. Each is tested on its own in tests/test_train.py.
# --------------------------------------------------------------------------- #

def lm_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """logits [B, T, V], targets [B, T] -> scalar mean next-token cross-entropy (nats)."""
    raise NotImplementedError


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW, betas (0.9, 0.95). Two param groups:
    decay    — every parameter with ndim >= 2 (matrices, embeddings)
    no-decay — every parameter with ndim < 2 (biases, LayerNorm w/b)
    Be ready to explain why decaying LayerNorm gains is a bad idea.
    """
    raise NotImplementedError


def lr_at(step: int, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup from 0 to max_lr over warmup_steps (lr_at(0) == 0 is fine),
    then cosine decay from max_lr to min_lr, reaching min_lr at total_steps and staying there.
    """
    raise NotImplementedError


def global_grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients concatenated (what clip_grad_norm_ measures). Ignore params with grad None."""
    raise NotImplementedError


def update_ratio(before: list[Tensor], model: nn.Module) -> float:
    """||theta_after - theta_before|| / ||theta_before||, over all params, as one number.

    `before` is [p.detach().clone() for p in model.parameters()] taken just before
    optimizer.step(). A healthy run sits around 1e-3; much larger means the lr is
    too high, much smaller means the model has almost stopped moving.
    """
    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model: GPT, data: Tensor, block_size: int, batch_size: int, n_batches: int, generator: torch.Generator) -> float:
    """Mean lm_loss over n_batches random batches. Remember model.eval() / model.train()."""
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Provided: metrics logging and reference numbers.
# --------------------------------------------------------------------------- #

class MetricsLogger:
    FIELDS = ["step", "lr", "train_loss", "val_loss", "grad_norm", "update_ratio", "tok_per_s"]

    def __init__(self, run_dir: Path | None):
        self.run_dir = run_dir
        self.rows: list[dict] = []
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._f = open(run_dir / "metrics.csv", "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
            self._w.writeheader()

    def log(self, **row) -> None:
        self.rows.append(row)
        if self.run_dir is not None:
            self._w.writerow({k: row.get(k, "") for k in self.FIELDS})
            self._f.flush()


def reference_losses(train_data: Tensor, d_vocab: int) -> dict[str, float]:
    counts = torch.bincount(train_data, minlength=d_vocab).float()
    p = counts / counts.sum()
    unigram = -(p[p > 0] * p[p > 0].log()).sum().item()
    return {"uniform ln(V)": math.log(d_vocab), "unigram entropy": unigram}


# --------------------------------------------------------------------------- #
# The loop. You write it.
# --------------------------------------------------------------------------- #

def train(model: GPT, train_data: Tensor, val_data: Tensor, tcfg: TrainConfig, logger: MetricsLogger) -> GPT:
    """Train `model` for tcfg.steps steps. Each step, in this order:

      1. set the lr for this step on every param group (lr_at)
      2. sample a batch with get_batch(..., block_size=model.cfg.n_ctx, generator=g)
      3. forward, lm_loss
      4. zero grads, backward
      5. measure global_grad_norm (before clipping — you want to see the raw value)
      6. clip to tcfg.grad_clip
      7. snapshot params if this is a log step, optimizer.step(), compute update_ratio
      8. every log_interval steps: logger.log(step, lr, train_loss, grad_norm, update_ratio, tok_per_s)
         every eval_interval steps (and the final step): also val_loss via estimate_loss

    Print a line at each eval so you can watch it. Use one torch.Generator seeded
    from tcfg.seed for batches and a separate one for eval batches.
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--d-model", type=int, default=Config.d_model)
    ap.add_argument("--n-layers", type=int, default=Config.n_layers)
    ap.add_argument("--n-heads", type=int, default=Config.n_heads)
    ap.add_argument("--n-ctx", type=int, default=Config.n_ctx)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok, train_data, val_data = load_shakespeare()
    cfg = Config(d_vocab=tok.d_vocab, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_mlp=4 * args.d_model, n_ctx=args.n_ctx)
    tcfg = TrainConfig(steps=args.steps)
    for k, v in reference_losses(train_data, tok.d_vocab).items():
        print(f"reference  {k:16s} {v:.3f}")

    model = GPT(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    run_dir = Path("runs") / args.name
    logger = MetricsLogger(run_dir)
    t0 = time.time()
    train(model, train_data, val_data, tcfg, logger)
    print(f"done in {time.time() - t0:.0f}s")

    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "chars": tok.chars}, run_dir / "model.pt")


if __name__ == "__main__":
    main()
