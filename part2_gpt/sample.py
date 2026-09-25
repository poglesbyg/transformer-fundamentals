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
