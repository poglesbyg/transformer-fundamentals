import torch

from part2_gpt.model import GPT, Config
from part4_induction.induction import BOS, ablate_head, induction_score, per_position_loss, prev_token_score, repeated_random_tokens


def test_repeated_random_tokens():
    toks = repeated_random_tokens(5, 7, 30, torch.Generator().manual_seed(0))
    assert toks.shape == (5, 15) and toks.dtype == torch.long
    assert (toks[:, 0] == BOS).all()
    assert torch.equal(toks[:, 1:8], toks[:, 8:])
    assert (toks[:, 1:] >= 1).all() and (toks[:, 1:] < 30).all()


def _pattern_with_offset(T, offset, H=3, hot_head=1):
    """Uniform causal attention everywhere, except `hot_head` attends exactly `offset` back when it can."""
    p = torch.tril(torch.ones(T, T))
    p = p / p.sum(-1, keepdim=True)
    p = p.expand(2, H, T, T).clone()
    p[:, hot_head] = 0
    for q in range(T):
        p[:, hot_head, q, max(q - offset, 0)] = 1
    return p


def test_prev_token_score():
    s = prev_token_score(_pattern_with_offset(9, 1))
    assert s.shape == (3,)
    assert s[1].item() == 1.0
    assert s[0] < 0.5 and s[2] < 0.5


def test_induction_score():
    seq_len = 6
    T = 1 + 2 * seq_len
    s = induction_score(_pattern_with_offset(T, seq_len - 1), seq_len)
    assert s.shape == (3,)
    assert s[1].item() == 1.0
    assert s[0] < 0.5
    assert induction_score(_pattern_with_offset(T, 1), seq_len)[1] == 0.0, "a prev-token head is not an induction head"


def test_per_position_loss_and_ablation():
    torch.manual_seed(0)
    cfg = Config(d_vocab=20, d_model=32, n_layers=2, n_heads=4, n_ctx=21, attn_only=True)
    m = GPT(cfg)
    toks = repeated_random_tokens(4, 10, 20, torch.Generator().manual_seed(0))
    loss = per_position_loss(m, toks)
    assert loss.shape == (20,)
    logits = m(toks)
    ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 20), toks[:, 1:].reshape(-1), reduction="none").view(4, 20).mean(0)
    torch.testing.assert_close(loss, ref)

    W_O = m.blocks[1].attn.W_O
    old = ablate_head(m, 1, 2)
    assert (W_O[2] == 0).all() and not (W_O[1] == 0).all()
    with torch.no_grad():
        W_O[2] = old
    torch.testing.assert_close(m(toks), logits)
