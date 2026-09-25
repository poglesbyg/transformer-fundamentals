"""Tests for part2_gpt/model.py.

Order matters: work top to bottom. Each test names the property it checks,
which is also the thing to say out loud in an interview.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from part2_gpt.model import GPT, MLP, Attention, Block, Config, Embed, LayerNorm, PosEmbed, count_params

CFG = Config(d_vocab=37, d_model=32, n_layers=2, n_heads=4, d_mlp=64, n_ctx=16)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _randomize(module):
    """Tests shouldn't pass just because biases are initialised to zero."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return module


# ---------------------------------------------------------------- LayerNorm

def test_layernorm_matches_torch():
    ln = _randomize(LayerNorm(CFG))
    ref = torch.nn.LayerNorm(CFG.d_model, eps=CFG.layer_norm_eps)
    with torch.no_grad():
        ref.weight.copy_(ln.w)
        ref.bias.copy_(ln.b)
    x = torch.randn(3, 5, CFG.d_model) * 4 + 2
    torch.testing.assert_close(ln(x), ref(x))


def test_layernorm_init():
    ln = LayerNorm(CFG)
    assert torch.equal(ln.w, torch.ones(CFG.d_model)) and torch.equal(ln.b, torch.zeros(CFG.d_model))


# ---------------------------------------------------------------- embeddings

def test_embed_is_a_lookup():
    e = Embed(CFG)
    assert e.W_E.shape == (CFG.d_vocab, CFG.d_model)
    toks = torch.randint(0, CFG.d_vocab, (2, 7))
    torch.testing.assert_close(e(toks), e.W_E[toks])


def test_pos_embed_depends_only_on_position():
    pe = PosEmbed(CFG)
    assert pe.W_pos.shape == (CFG.n_ctx, CFG.d_model)
    a = pe(torch.randint(0, CFG.d_vocab, (2, 5)))
    b = pe(torch.randint(0, CFG.d_vocab, (2, 5)))
    assert a.shape == (2, 5, CFG.d_model)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[0], pe.W_pos[:5])


# ---------------------------------------------------------------- Attention

def test_attention_param_shapes():
    a = Attention(CFG)
    H, D, Dh = CFG.n_heads, CFG.d_model, CFG.d_head
    for name in ["W_Q", "W_K", "W_V"]:
        assert getattr(a, name).shape == (H, D, Dh), name
    for name in ["b_Q", "b_K", "b_V"]:
        assert getattr(a, name).shape == (H, Dh), name
    assert a.W_O.shape == (H, Dh, D)
    assert a.b_O.shape == (D,)


def _reference_attention(a: Attention, x):
    """Same weights, computed via torch's fused attention kernel."""
    q = torch.einsum("btd,hde->bhte", x, a.W_Q) + a.b_Q[None, :, None]
    k = torch.einsum("btd,hde->bhte", x, a.W_K) + a.b_K[None, :, None]
    v = torch.einsum("btd,hde->bhte", x, a.W_V) + a.b_V[None, :, None]
    z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return torch.einsum("bhte,hed->btd", z, a.W_O) + a.b_O


def test_attention_matches_sdpa():
    a = _randomize(Attention(CFG))
    x = torch.randn(2, 9, CFG.d_model)
    torch.testing.assert_close(a(x), _reference_attention(a, x), atol=1e-5, rtol=1e-4)


def test_attention_pattern_is_causal_and_normalised():
    a = _randomize(Attention(CFG))
    a(torch.randn(2, 6, CFG.d_model))
    p = a.pattern
    assert p is not None and p.shape == (2, CFG.n_heads, 6, 6)
    assert not p.requires_grad, "store the pattern detached"
    torch.testing.assert_close(p.sum(-1), torch.ones(2, CFG.n_heads, 6))
    assert (p.triu(diagonal=1) == 0).all(), "a query attended to a future key"


def test_attention_gradients_reach_qk():
    """Matching the forward pass isn't enough: if the pattern is detached inside the
    computation, W_Q and W_K get no gradient and attention can never learn where to look."""
    a = _randomize(Attention(CFG))
    a(torch.randn(2, 6, CFG.d_model)).pow(2).sum().backward()
    for name in ["W_Q", "W_K", "b_Q", "b_K", "W_V", "W_O"]:
        g = getattr(a, name).grad
        assert g is not None and g.abs().sum() > 0, f"no gradient reaches {name}"


def test_attention_first_position_attends_only_to_itself():
    a = _randomize(Attention(CFG))
    a(torch.randn(1, 5, CFG.d_model))
    torch.testing.assert_close(a.pattern[0, :, 0, 0], torch.ones(CFG.n_heads))


# ---------------------------------------------------------------- MLP / Block

def test_mlp():
    m = _randomize(MLP(CFG))
    assert m.W_in.shape == (CFG.d_model, CFG.d_mlp) and m.W_out.shape == (CFG.d_mlp, CFG.d_model)
    x = torch.randn(2, 3, CFG.d_model)
    torch.testing.assert_close(m(x), F.gelu(x @ m.W_in + m.b_in, approximate="tanh") @ m.W_out + m.b_out)


def test_block_is_residual():
    """With attention and MLP outputs zeroed, a pre-LN block must be the identity."""
    blk = _randomize(Block(CFG))
    with torch.no_grad():
        for p in [blk.attn.W_O, blk.attn.b_O, blk.mlp.W_out, blk.mlp.b_out]:
            p.zero_()
    x = torch.randn(2, 5, CFG.d_model)
    torch.testing.assert_close(blk(x), x)


def test_attn_only_block_has_no_mlp():
    blk = Block(Config(**{**CFG.__dict__, "attn_only": True}))
    assert not hasattr(blk, "mlp") or blk.mlp is None
    assert blk(torch.randn(1, 4, CFG.d_model)).shape == (1, 4, CFG.d_model)


# ---------------------------------------------------------------- GPT

def test_gpt_shapes():
    m = GPT(CFG)
    assert m(torch.randint(0, CFG.d_vocab, (3, 11))).shape == (3, 11, CFG.d_vocab)


def test_gpt_rejects_too_long_context():
    with pytest.raises(AssertionError):
        GPT(CFG)(torch.randint(0, CFG.d_vocab, (1, CFG.n_ctx + 1)))


def test_gpt_is_causal():
    """Changing token t must not change the logits at any position < t."""
    m = _randomize(GPT(CFG))
    toks = torch.randint(0, CFG.d_vocab, (1, 12))
    base = m(toks)
    t = 7
    toks2 = toks.clone()
    toks2[0, t] = (toks2[0, t] + 1) % CFG.d_vocab
    out = m(toks2)
    torch.testing.assert_close(out[:, :t], base[:, :t])
    assert not torch.allclose(out[:, t:], base[:, t:]), "the change should be visible from position t onward"


def test_gpt_init_loss_is_uniform():
    """With std-0.02 init the logits are ~0, so the loss must be ~ln(d_vocab).
    If it isn't, something in your init or forward is off before training even starts."""
    cfg = Config(d_vocab=65, d_model=64, n_layers=2, n_heads=4, d_mlp=256, n_ctx=32)
    m = GPT(cfg)
    toks = torch.randint(0, cfg.d_vocab, (8, 32))
    logits = m(toks)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.d_vocab), toks[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(cfg.d_vocab)) < 0.05


def test_count_params():
    for cfg in [CFG, Config(), Config(**{**CFG.__dict__, "attn_only": True})]:
        assert count_params(cfg) == sum(p.numel() for p in GPT(cfg).parameters())


def test_count_params_gpt2_small():
    gpt2 = Config(d_vocab=50257, d_model=768, n_layers=12, n_heads=12, d_mlp=3072, n_ctx=1024)
    # GPT-2 small is 124,439,808. It ties W_U to W_E and has no b_U; this model has both,
    # which adds 50257 * 768 + 50257 = 38,647,633.
    assert count_params(gpt2) == 163_087_441
