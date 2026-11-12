"""ktt_flash reading q / k / v in place from [B, S, C] bases (heads side by side, no
split copies; ktt::_debug_flash_pitched): one program, results against float64 within
test_flash's bound, for a fused qkv base, separate bases with grouped heads, a ragged S
and causal."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

BOUND = 8e-3  # test_flash.BOUND


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _heads(base, col, h, d):
    b, s, _ = base.shape
    return base[:, :, col : col + h * d].reshape(b, s, h, d).transpose(1, 2).double()


# (b, s, h, hk, d, fused, causal)
CASES = {
    "fused-300": (2, 300, 4, 4, 64, True, False),
    "gqa-128": (1, 128, 8, 2, 64, False, False),
    "fused-causal-77": (1, 77, 2, 2, 128, True, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_pitched_matches_reference(native, case):
    b, s, h, hk, d, fused, causal = CASES[case]
    g = torch.Generator().manual_seed(s + d)
    if fused:
        qkv = torch.randn(b, s, 3 * h * d, generator=g).bfloat16()
        qb = kb = vb = qkv
        cols = [0, h * d, 2 * h * d]
    else:
        qb = torch.randn(b, s, h * d, generator=g).bfloat16()
        kb, vb = (torch.randn(b, s, hk * d, generator=g).bfloat16() for _ in range(2))
        cols = [0, 0, 0]
    q, k, v = (_heads(t, c, n, d) for t, c, n in zip((qb, kb, vb), cols, (h, hk, hk)))
    rep = h // hk
    ref = F.scaled_dot_product_attention(
        q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=causal
    )
    ins = [qkv.to("tt")] * 3 if fused else [t.to("tt") for t in (qb, kb, vb)]
    torch.tt.synchronize()
    before = kohakutt.counters()
    y = torch.ops.ktt._debug_flash_pitched(*ins, h, hk, d, cols, causal)
    torch.tt.synchronize()
    after = kohakutt.counters()
    assert after["programs"] - before["programs"] == 1
    assert _rel(y.cpu(), ref) <= BOUND
