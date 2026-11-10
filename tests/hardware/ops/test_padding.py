"""Tile padding that holds inf (a slice of a larger tensor keeps the tile data) must not
reach a result: attention (output and gradients, dense and packed) and softmax /
log_softmax / cross_entropy over a last dim that is not a multiple of 32 equal the
clean-padding run."""

import pytest
import torch
import torch.nn.functional as F


def _values(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen).to(torch.bfloat16)


def _dirty(x):
    """x on tt whose tile padding (last two dims) holds inf."""
    *lead, r, c = x.shape
    big = torch.full(
        (*lead, (r + 31) // 32 * 32 + 32, (c + 31) // 32 * 32 + 32),
        float("inf"),
        dtype=x.dtype,
    )
    big[..., :r, :c] = x
    return big.to("tt")[..., :r, :c]


DIRTY = [{0}, {1}, {2}, {1, 2}]
DIRTY_IDS = ["q", "k", "v", "kv"]
# (shape, causal): a last key tile of 1 key, also causal; ViT's 197 tokens.
TRAIN_CASES = [
    ((2, 3, 33, 32), False),
    ((2, 3, 33, 32), True),
    ((1, 12, 197, 64), False),
]


def _train(native, dirty, shape=(2, 3, 33, 32), causal=False):
    data = [_values(shape, 80 + i) for i in range(3)]
    ins = [
        (_dirty(x) if i in dirty else x.to("tt")).detach().requires_grad_(True)
        for i, x in enumerate(data)
    ]
    out = F.scaled_dot_product_attention(*ins, is_causal=causal)
    g = _values(shape, 99)
    out.backward(_dirty(g) if 3 in dirty else g.to("tt"))
    return [out.detach().cpu()] + [t.grad.cpu() for t in ins]


@pytest.mark.parametrize("shape,causal", TRAIN_CASES, ids=["33", "33-causal", "197"])
@pytest.mark.parametrize("dirty", DIRTY + [{3}], ids=DIRTY_IDS + ["grad"])
def test_training_attention_ignores_padding(native, dirty, shape, causal):
    """Output and dq, dk, dv with inf in the padding of q / k / v / the output gradient
    equal the clean run."""
    clean = _train(native, set(), shape, causal)
    got = _train(native, dirty, shape, causal)
    for name, c, g in zip(["out", "dq", "dk", "dv"], clean, got):
        assert torch.equal(c, g), name


# (q shape, k / v shape, causal, attention mask, KTT_FLASH): ktt_flash cases, then
# ktt_attention (ktt_flash off; a head dim off the tile, which ktt_flash does not take).
INFER_CASES = {
    "flash": ((2, 3, 33, 32), (2, 3, 33, 32), False, False, "1"),
    "flash-causal": ((2, 3, 70, 64), (2, 3, 70, 64), True, False, "1"),
    "flash-mask": ((2, 3, 40, 32), (2, 3, 33, 32), False, True, "1"),
    "flash-197": ((1, 12, 197, 64), (1, 12, 197, 64), False, False, "1"),
    "own-causal": ((2, 3, 70, 64), (2, 3, 70, 64), True, False, "0"),
    "own-d40": ((2, 3, 33, 40), (2, 3, 33, 40), True, False, "1"),
}


def _infer(shapes, dirty):
    qs, ks, causal, masked, _ = shapes
    data = [_values(s, 60 + i) for i, s in enumerate([qs, ks, ks])]
    ins = [_dirty(x) if i in dirty else x.to("tt") for i, x in enumerate(data)]
    mask = None
    if masked:
        mask = _values((1, 1, qs[2], ks[2]), 7).clamp(-2, 2).to("tt")
    with torch.no_grad():
        out = F.scaled_dot_product_attention(*ins, attn_mask=mask, is_causal=causal)
    return out.cpu()


@pytest.mark.parametrize("case", list(INFER_CASES))
@pytest.mark.parametrize("dirty", DIRTY, ids=DIRTY_IDS)
def test_attention_ignores_padding(native, monkeypatch, case, dirty):
    """Inference attention with inf in the padding of q / k / v equals the clean run."""
    monkeypatch.setenv("KTT_FLASH", INFER_CASES[case][4])
    clean = _infer(INFER_CASES[case], set())
    assert torch.isfinite(clean).all()
    assert torch.equal(_infer(INFER_CASES[case], dirty), clean)


def _packed_dirty(x):
    """x [T, H, D] on tt as a view of [T, H D] whose padding rows past T hold inf."""
    t, h, d = x.shape
    big = torch.full(((t + 31) // 32 * 32 + 32, h * d), float("inf"), dtype=x.dtype)
    big[:t] = x.reshape(t, h * d)
    return big.to("tt")[:t].view(t, h, d)


def _varlen(native, dirty, causal):
    cu = [0, 40, 47, 137]
    data = [_values((137, 4, 64), 40 + i) for i in range(4)]
    ins = [
        (_packed_dirty(x) if i in dirty else x.to("tt")).detach().requires_grad_(True)
        for i, x in enumerate(data[:3])
    ]
    out = native.sdpa_varlen(*ins, cu, is_causal=causal)
    g = _packed_dirty(data[3]) if 3 in dirty else data[3].to("tt")
    grads = torch.autograd.grad(out, ins, g)
    return [out.detach().cpu()] + [t.cpu() for t in grads]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dirty", DIRTY + [{3}], ids=DIRTY_IDS + ["grad"])
def test_varlen_attention_ignores_padding(native, dirty, causal):
    """Packed attention (137 tokens: padding rows in the last tile) with inf there: the
    output and dq, dk, dv equal the clean run."""
    clean = _varlen(native, set(), causal)
    got = _varlen(native, dirty, causal)
    for name, c, g in zip(["out", "dq", "dk", "dv"], clean, got):
        assert torch.equal(c, g), name


@pytest.mark.parametrize("shape", [(33, 70), (4, 33, 70)])
def test_softmax_ignores_padding(native, shape):
    """softmax, log_softmax over the ragged last dim equal the clean run."""
    x = _values(shape, 3)
    for fn in (lambda t: F.softmax(t, -1), lambda t: F.log_softmax(t, -1)):
        assert torch.equal(fn(_dirty(x)).cpu(), fn(x.to("tt")).cpu())


def test_cross_entropy_ignores_padding(native):
    """Cross entropy of logits [33, 70] with inf in the padding equals the clean run."""
    x = _values((33, 70), 4)
    target = torch.randint(0, 70, (33,), generator=torch.Generator().manual_seed(5))
    clean = F.cross_entropy(x.to("tt"), target.to("tt"))
    got = F.cross_entropy(_dirty(x), target.to("tt"))
    assert torch.equal(clean.cpu(), got.cpu())
