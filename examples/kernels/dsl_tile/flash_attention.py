"""Flash attention (forward) at the tile level: per query block, a loop over key blocks
carrying the running maximum, sum and acc = acc * alpha + p @ v; out = acc / sum.
Causal or not; compared with float64, the backend's flash kernel and TT-NN's SDPA.
Usage: python flash_attention.py [--shape B H S D] [--causal]"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
from common import timed


def exp_neg(x):
    """e^x for x <= 0 written with vector-unit arithmetic: (Taylor of x / 64)^64,
    x clamped at -88 (e^-88 is below fp32's normal range)."""
    y = kd.maximum(x, -88.0) * (1 / 64)
    p = 1 + y * (1 + y * (1 / 2 + y * (1 / 6 + y * (1 / 24))))
    for _ in range(6):
        p = p * p
    return p


@kd.kernel
def flash_attention(
    q: kd.Tensor,
    k: kd.Tensor,
    v: kd.Tensor,
    masks: kd.Tensor,
    out: kd.Out,
    scale: kd.Runtime[float],
    rows: kd.Static[int] = 8,
    keys: kd.Static[int] = 8,
    causal: kd.Static[int] = 0,
    poly: kd.Static[int] = 1,
):
    """softmax(q k^T scale) v per head; q [B * H, S, D], k and v [B * Hk, Sk, D]
    (H a multiple of Hk); `masks` = kd.score_masks(Sk); causal needs S == Sk.
    `poly`: the exponentials by `exp_neg` (else the vector unit's accurate exp)."""
    exp = exp_neg if poly else kd.exp
    qt, kt = kd.num_tiles(q, -2) // kd.size(q, 0), kd.num_tiles(k, -2) // kd.size(k, 0)
    blocks = kd.cdiv(qt, rows)
    group = kd.size(q, 0) // kd.size(k, 0)
    for u in kd.grid(kd.size(q, 0) * blocks):
        head, b = u // blocks, u % blocks
        first = head * qt + b * rows
        n = kd.minimum(rows, qt - b * rows)
        qb = kd.load(q.tiles[first : first + n, :])
        m = kd.carry(-1e30)
        den = kd.carry(0.0)
        acc = kd.carry(0.0)
        # Causal: key tiles up to the block's last query tile.
        seen = kd.minimum(kt, b * rows + n) if causal else kt
        for j in kd.range(kd.cdiv(seen, keys)):
            k0 = head // group * kt + j * keys
            nk = kd.minimum(keys, seen - j * keys)
            kb = kd.load(k.tiles[k0 : k0 + nk, :])
            vb = kd.load(v.tiles[k0 : k0 + nk, :])
            s = kd.mask_scores(
                (qb @ kb.T) * scale, masks, b * rows, j * keys, kd.size(k, -2), causal
            )
            m_new = kd.maximum(m, kd.reduce_max(s))
            p = exp(s - m_new.col())
            alpha = exp(m - m_new)
            den.next(den * alpha + kd.reduce_sum(p))
            acc.next(acc * alpha.col() + p @ vb)
            m.next(m_new)
        kd.store(out.tiles[first : first + n, :], (acc / den.col()).cast(out.dtype))


def query_rows(heads, q_tiles, cores=110, most=16):
    """Query tile rows per unit: the fewest rows on the busiest core, then the most
    rows per unit (fewer k and v reads)."""

    def busiest(r):
        return -(-heads * -(-q_tiles // r) // cores) * min(r, q_tiles)

    return min(range(most, 1, -1), key=busiest)


def attention(q, k, v, causal=False, rows=None, keys=8, poly=1):
    """The tile kernel on [B, H, S, D] tensors (k, v [B, Hk, Sk, D])."""
    b, h, s, d = q.shape
    if rows is None and causal:
        rows = 8  # measured faster than the balanced choice (uneven blocks)
    if rows is None:
        cols, grid_rows = torch.ops.ktt.compute_grid()
        rows = query_rows(b * h, -(-s // 32), cols * grid_rows)
    masks = _masks(k.shape[-2], q.device)
    out = flash_attention(
        q.reshape(b * h, s, d),
        k.reshape(-1, k.shape[-2], d),
        v.reshape(-1, k.shape[-2], d),
        masks,
        1 / math.sqrt(d),
        rows=rows,
        keys=keys,
        causal=int(causal),
        poly=poly,
    )
    return out.reshape(b, h, s, d)


_MASKS = {}


def _masks(keys, device):
    key = (keys, str(device))
    if key not in _MASKS:
        _MASKS[key] = kd.score_masks(keys, device)
    return _MASKS[key]


def inputs(shape, kv_heads=None, seed=0):
    """q, k, v (bf16) on the open device; k and v with `kv_heads` heads."""
    g = torch.Generator().manual_seed(seed)
    b, h, s, d = shape
    hk = kv_heads or h
    dev = kohakutt.device(0)
    q = torch.randn(b, h, s, d, generator=g).bfloat16()
    k = torch.randn(b, hk, s, d, generator=g).bfloat16()
    v = torch.randn(b, hk, s, d, generator=g).bfloat16()
    return q.to(dev), k.to(dev), v.to(dev)


def reference(q, k, v, causal=False):
    """float64 attention on the CPU (grouped heads repeated)."""
    q, k, v = (x.cpu().double() for x in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def relative_rms(got, want):
    """RMS of the error relative to the RMS of `want`."""
    diff = got.cpu().double() - want
    return (diff.pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()


def compare(shape, causal=False, kv_heads=None):
    """Relative RMS errors against float64: the tile kernel and the backend's SDPA."""
    q, k, v = inputs(shape, kv_heads)
    want = reference(q, k, v, causal)
    rep = shape[1] // k.shape[1]
    kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    return {
        "tile": relative_rms(attention(q, k, v, causal), want),
        "tile, accurate exp": relative_rms(attention(q, k, v, causal, poly=0), want),
        "backend": relative_rms(
            F.scaled_dot_product_attention(q, kk, vv, is_causal=causal), want
        ),
    }


def bench(shape, causal=False):
    """{name: seconds}: the tile kernel, the backend's flash kernel, TT-NN's SDPA."""
    q, k, v = inputs(shape)
    out = {
        "tile": timed(lambda: attention(q, k, v, causal), reps=5, rounds=3),
        "tile, accurate exp": timed(
            lambda: attention(q, k, v, causal, poly=0), reps=5, rounds=3
        ),
    }
    for name in ("flash", "ttnn"):
        try:
            with kohakutt.sdpa_kernel(name):
                out[name] = timed(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                    reps=5,
                    rounds=3,
                )
        except RuntimeError as e:  # a route that does not take the call
            out[name] = f"n/a ({type(e).__name__})"
    return out


def main():
    """Runs attention at one shape; prints errors, times and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=(1, 8, 1024, 64))
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    shape = tuple(args.shape)
    try:
        for key, value in {
            **compare(shape, args.causal),
            **bench(shape, args.causal),
        }.items():
            print(f"  {key}: {value}")
        q, k, v = inputs(shape)
        b, h, s, d = shape
        print(
            flash_attention.plan(
                q.reshape(b * h, s, d),
                k.reshape(b * h, s, d),
                v.reshape(b * h, s, d),
                _masks(s, q.device),
                1 / math.sqrt(d),
                causal=int(args.causal),
            ).report()
        )
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
