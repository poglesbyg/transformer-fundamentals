"""Tiny Shakespeare, character level. ~1.1M characters, 65-symbol vocabulary.

Character level is deliberate: the vocabulary is small enough that you can
reason about every reference loss by hand (see REFERENCE_LOSSES in train.py),
and a CPU trains a model that produces recognisable Shakespeare in minutes.
"""

from pathlib import Path
import urllib.request

import torch
from torch import Tensor

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_text() -> str:
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text()


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def d_vocab(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: Tensor | list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def train_val_split(data: Tensor, val_frac: float = 0.1) -> tuple[Tensor, Tensor]:
    """Contiguous split: the last 10% is validation. (Why must this not be a random shuffle of positions?)"""
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def get_batch(data: Tensor, block_size: int, batch_size: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Sample `batch_size` random windows from the 1-D token tensor `data`.

    -> x [batch_size, block_size], y [batch_size, block_size], where y is x shifted
    left by one: y[b, t] is the token that follows x[b, t] in `data`.

    Every valid start index must be reachable, and no window may run off the end.
    This function is where the most common silent bug in language modelling lives.
    """
    # Randomly sample start indices for the batch
    start_indices = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    # Create the input and target batches
    x = torch.stack([data[i:i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_indices])
    return x, y
    raise NotImplementedError


def load_shakespeare() -> tuple[CharTokenizer, Tensor, Tensor]:
    text = load_text()
    tok = CharTokenizer(text)
    train, val = train_val_split(tok.encode(text))
    return tok, train, val
