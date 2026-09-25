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

BOS = 0


def repeated_random_tokens(batch: int, seq_len: int, d_vocab: int, generator: torch.Generator) -> Tensor:
    """-> [batch, 1 + 2*seq_len] long: BOS, then seq_len random tokens from [1, d_vocab), then the same seq_len again."""
    raise NotImplementedError


def prev_token_score(pattern: Tensor) -> Tensor:
    """pattern [B, H, T, T] -> [H]: mean attention from each query position q >= 1 to key q-1."""
    raise NotImplementedError


def induction_score(pattern: Tensor, seq_len: int) -> Tensor:
    """pattern [B, H, T, T] from repeated_random_tokens input (T = 1 + 2*seq_len) -> [H].

    For each query in the second copy, the induction target is the key one
    position *after* the same token's earlier occurrence: the offset is seq_len - 1
    back from the query. Average the attention on that diagonal over queries in the second copy.
    """
    raise NotImplementedError


def per_position_loss(model: GPT, tokens: Tensor) -> Tensor:
    """tokens [B, T] -> [T-1]: next-token cross-entropy at each position, averaged over batch."""
    raise NotImplementedError


@torch.no_grad()
def ablate_head(model: GPT, layer: int, head: int) -> Tensor:
    """Zero W_O[head] of the given layer in place; return the old slice so you can restore it."""
    raise NotImplementedError


def train_induction_model(steps: int = 3000, min_len: int = 10, max_len: int = 30, d_vocab: int = 64, fixed_len: int | None = None, seed: int = 0) -> GPT:
    """Train an attn-only 2-layer model on repeated_random_tokens. You write the loop
    (reuse configure_optimizer from part2_gpt.train if you like).
    Suggested config: d_model 64, n_heads 4, n_ctx = 1 + 2*max_len, lr 1e-3, batch 64.

    Each step, draw seq_len uniformly from [min_len, max_len], or use fixed_len if given.
    Log loss on the first and second halves separately. You should see a plateau and
    then a sudden drop in second-half loss: that drop is the induction head forming.

    Do this first, before the varying-length version: train with fixed_len=20 and
    run the head scores. The second-half loss still goes to ~0, but look at *which
    layer* has the high induction score, and where the prev-token head went. Work out
    what the model learned instead and why a fixed seq_len allowed it. (Hint: learned
    absolute position embeddings.) Checking that your task doesn't admit a shortcut
    is the same habit you'll need for every ARENA circuit claim.

    Even with varying lengths, some seeds settle on a different solution (loss low,
    induction scores low everywhere). If that happens, don't just reseed: look at
    where the heads actually attend (pattern[:, h, q].topk) and describe the
    alternative. Then reseed. With the defaults, most seeds give a layer-0 head
    with prev_token_score ~0.6+ and layer-1 heads with induction_score ~0.8.
    """
    raise NotImplementedError


def main() -> None:
    seq_len, d_vocab = 20, 64  # evaluation length; training varies it
    model = train_induction_model(d_vocab=d_vocab)
    g = torch.Generator().manual_seed(123)
    toks = repeated_random_tokens(32, seq_len, d_vocab, g)

    loss = per_position_loss(model, toks)
    print(f"ln(d_vocab) = {math.log(d_vocab):.2f}")
    print(f"first half loss  {loss[1:seq_len].mean():.3f}")
    print(f"second half loss {loss[seq_len + 1:].mean():.3f}")

    model(toks)
    for layer, block in enumerate(model.blocks):
        p = block.attn.pattern
        print(f"layer {layer}  prev-token {prev_token_score(p).numpy().round(2)}  induction {induction_score(p, seq_len).numpy().round(2)}")

    # TODO: ablate the top induction head, re-measure second-half loss, restore it.
    # Then ablate the top prev-token head in layer 0 and do the same. Explain both results,
    # including why ablating one of several induction heads may hurt less than you'd expect.


if __name__ == "__main__":
    main()
