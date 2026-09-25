"""Autoregressive sampling.

    uv run python -m part2_gpt.sample runs/baseline --prompt "ROMEO:" --temperature 0.8 --top-k 20
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor

from part2_gpt.data import CharTokenizer
from part2_gpt.model import GPT, Config


@torch.no_grad()
def sample(
    model: GPT,
    prompt: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """prompt: [T0] long -> [T0 + max_new_tokens] long.

    Each step: crop the context to the last n_ctx tokens, forward, take the last
    position's logits, and pick the next token:
      temperature == 0  -> greedy argmax
      otherwise         -> divide logits by temperature; if top_k, set all but the
                           top_k logits to -inf; sample with torch.multinomial(generator=...)
    Put the model in eval mode.

    No KV cache here: every step recomputes the whole context. Know what a KV cache
    would store, its memory in bytes for GPT-2 small at 1024 tokens, and why it
    makes generation O(T) per token instead of O(T^2).
    """
    model.eval()
    toks = prompt.clone()                                   # [T]
    for _ in range(max_new_tokens):
        ctx = toks[-model.cfg.n_ctx:]                       # crop to the context window
        logits = model(ctx[None])[0, -1]                    # [1, T] -> [1, T, V] -> [V]: last position only
        if temperature == 0:
            nxt = logits.argmax()[None]                     # greedy
        else:
            logits = logits / temperature                   # <1 sharpens, >1 flattens
            if top_k is not None:
                kth = logits.topk(top_k).values[-1]         # k-th largest logit
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        toks = torch.cat([toks, nxt])                       # [T+1]
    return toks
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.run / "model.pt")
    model = GPT(Config(**ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer("".join(ckpt["chars"]))
    out = sample(model, tok.encode(args.prompt), args.tokens, temperature=args.temperature, top_k=args.top_k, generator=torch.Generator().manual_seed(args.seed))
    print(tok.decode(out))


if __name__ == "__main__":
    main()
