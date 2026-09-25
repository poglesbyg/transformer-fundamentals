"""Find an induction head in a model you trained yourself.

This is the bridge between building transformers and ARENA's interpretability
material. ARENA shows you induction heads in a pretrained model. Here you train
a 2-layer attention-only model on sequences of random tokens that repeat once:

    [BOS] t1 t2 ... tN t1 t2 ... tN

The first half is unpredictable (the best possible loss there is ln(d_vocab)). The second half is
perfectly predictable, but only if the model learns the two-head circuit:
  layer 0: a previous-token head writes "the token before me was A" into position i
  layer 1: an induction head at the current token A looks for earlier positions whose
           previous token was A, attends to them, and copies the token found there

Run:  uv run pytest tests/test_induction.py -q
Then: uv run python -m part4_induction.induction

What "done" looks like, and what to be able to explain without notes:
  1. Per-position loss: first half ~ln(d_vocab), second half near 0. Say why each.
  2. A layer-0 head with a high prev_token_score and a layer-1 head with a high
     induction_score. Draw both of their attention patterns from memory.
  3. Ablating the induction head (zero its W_O slice) sends second-half loss back
     toward ln(d_vocab). Ablating an unrelated head doesn't. That's the causal
     evidence; the attention scores alone are only correlational.
  4. Why can't a 1-layer attention-only model do this? (K-composition.)
"""

import math

import torch
from torch import Tensor

from part2_gpt.model import GPT, Config

import torch.nn.functional as F

BOS = 0


def repeated_random_tokens(batch: int, seq_len: int, d_vocab: int, generator: torch.Generator) -> Tensor:
    x = torch.randint(1, d_vocab, (batch, seq_len), generator=generator)     # [B, L], never BOS
    return torch.cat([torch.full((batch, 1), BOS), x, x], dim=1)              # [B, 1 + 2L]


def prev_token_score(pattern: Tensor) -> Tensor:
    # the diagonal one below the main one: attention from q to q-1, for every q >= 1
    return pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=(0, -1))    # [H]


def induction_score(pattern: Tensor, seq_len: int) -> Tensor:
    # query q attends to key q-(L-1); element i of this diagonal is (q=i+L-1, k=i)
    d = pattern.diagonal(offset=-(seq_len - 1), dim1=-2, dim2=-1)
    return d[..., 2:].mean(dim=(0, -1))    # keep only queries in the second copy (q >= L+1)


def per_position_loss(model: GPT, tokens: Tensor) -> Tensor:
    logits = model(tokens)                                                    # [B, T, V]
    return F.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none").mean(0)  # [T-1]


@torch.no_grad()
def ablate_head(model: GPT, layer: int, head: int) -> Tensor:
    W_O = model.blocks[layer].attn.W_O
    old = W_O[head].clone()
    W_O[head] = 0                          # this head now writes nothing into the residual stream
    return old


def train_induction_model(steps: int = 3000, min_len: int = 10, max_len: int = 30, d_vocab: int = 64, fixed_len: int | None = None, seed: int = 0) -> GPT:
    from part2_gpt.train import configure_optimizer
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = GPT(Config(d_vocab=d_vocab, d_model=64, n_layers=2, n_heads=4, n_ctx=1 + 2 * max_len, attn_only=True))
    opt = configure_optimizer(model, 1e-3, 0.01)
    for step in range(steps):
        L = fixed_len or int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        loss = per_position_loss(model, repeated_random_tokens(64, L, d_vocab, g))
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        if step % 250 == 0:
            print(f"step {step:4d}  L={L:2d}  1st half {loss[1:L].mean():.3f}  2nd half {loss[L + 1:].mean():.3f}")
    return model


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seq_len, d_vocab = 20, 64
    model = train_induction_model(d_vocab=d_vocab, fixed_len=args.fixed_len, seed=args.seed)
    toks = repeated_random_tokens(32, seq_len, d_vocab, torch.Generator().manual_seed(123))

    def second_half() -> float:
        return per_position_loss(model, toks)[seq_len + 1:].mean().item()

    loss = per_position_loss(model, toks)
    print(f"\nln(d_vocab) = {math.log(d_vocab):.2f}   1st half {loss[1:seq_len].mean():.3f}   2nd half {second_half():.3f}")
    model(toks)
    for layer, block in enumerate(model.blocks):
        p = block.attn.pattern
        print(f"layer {layer}  prev-token {prev_token_score(p).numpy().round(2)}  induction {induction_score(p, seq_len).numpy().round(2)}")

    for layer in range(2):
        for h in range(4):
            old = ablate_head(model, layer, h)
            print(f"ablate L{layer}H{h}: 2nd half {second_half():.3f}")
            with torch.no_grad():
                model.blocks[layer].attn.W_O[h] = old


if __name__ == "__main__":
    main()
