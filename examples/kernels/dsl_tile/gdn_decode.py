"""One decode step of a gated delta net (GDN) at the tile level: per value head the
state h [V, K] decays, takes the delta rule's update from the step's k and v, and
gives o = h q. Compared with float64 and the same step as eager torch ops.
Usage: python examples/kernels/dsl_tile/gdn_decode.py [--seqs N] [--heads HK HV]"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl as kd

sys.path.insert(0, str(Path(__file__).parent))
from common import timed

EPS = 1e-6


@kd.kernel
def gdn_decode(
    h: kd.Tensor,
    q: kd.Tensor,
    k: kd.Tensor,
    v: kd.Tensor,
    decay: kd.Tensor,
    beta: kd.Tensor,
    h_out: kd.Out,
    o: kd.Out,
    scale: kd.Runtime[float],
    rows: kd.Static[int] = 4,
    group: kd.Static[int] = 3,
):
    """h [N HV V, K] fp32 state rows; q, k [N HK 32, K]: head j's vector in row 32 j;
    v, decay, beta [N HV V, 1] fp32 per state row. h_out = h decay + u k^, o = h_out
    q^ scale with u = (v - decay h k^) beta, k^ = k / |k| (`rows` = V / 32 tile rows
    per head, `group` = HV / HK)."""
    for t in kd.grid(kd.num_tile_rows(h)):
        kr = kd.load(k.tiles[t // rows // group, :]).cast(kd.f32).row()
        qr = kd.load(q.tiles[t // rows // group, :]).cast(kd.f32).row()
        rk = kd.rsqrt(kd.reduce_sum(kr * kr, exact=True) + EPS)
        rq = kd.rsqrt(kd.reduce_sum(qr * qr, exact=True) + EPS) * scale
        kq = kd.reduce_sum(kr * qr, exact=True)
        row = kd.load(h.tiles[t, :])
        # Sums of two operands: the columns factored out of h k^ and h_out q^.
        hk = kd.reduce_sum(row * kr, exact=True)
        hq = kd.reduce_sum(row * qr, exact=True)
        d = kd.load(decay.tiles[t])
        u = (kd.load(v.tiles[t]) - d * rk * hk) * kd.load(beta.tiles[t]) * rk
        kd.store(h_out.tiles[t, :], row * d.col() + u.col() * kr)
        kd.store(o.tiles[t], rq * (d * hq + u * kq))


def _f(x):
    """x in fp32, or kept in float64 (the reference)."""
    return x if x.dtype == torch.float64 else x.float()


def gates(a, b, a_log, dt_bias):
    """(decay, beta) [N, HV]: exp(-exp(A_log) softplus(a + dt_bias)), sigmoid(b)."""
    g = -_f(a_log).exp() * F.softplus(_f(a) + _f(dt_bias))
    return g.exp(), _f(b).sigmoid()


def layouts(q, k, v, decay, beta, hk, hv, dk, dv):
    """The kernel's operands from the model's: q, k [N, HK K] -> [N HK 32, K] (each
    head's vector in the first row of its tile row); v, decay, beta -> one value per
    state row [N HV V, 1] fp32."""
    n = q.shape[0]

    def padded(x):
        return F.pad(x.reshape(n * hk, 1, dk), (0, 0, 0, 31)).reshape(n * hk * 32, dk)

    def per_row(x):
        return x.reshape(n * hv, 1).expand(n * hv, dv).reshape(n * hv * dv, 1)

    col = v.float().reshape(n * hv * dv, 1)
    return padded(q), padded(k), col, per_row(decay), per_row(beta)


def step_tile(h, q, k, v, a, b, a_log, dt_bias, shape, scale):
    """The tile kernel's step from the model's tensors: (h_out [N HV V, K], o [N,
    HV V]); h [N HV V, K] fp32."""
    n, hk, hv, dk, dv = shape
    assert dv % 32 == 0 and hv % hk == 0, "V is whole tiles, HV a multiple of HK"
    decay, beta = gates(a, b, a_log, dt_bias)
    ops = layouts(q, k, v, decay, beta, hk, hv, dk, dv)
    h_out, o = gdn_decode(h, *ops, scale, rows=dv // 32, group=hv // hk)
    return h_out, o.reshape(n, hv * dv)


def step_eager(h, q, k, v, a, b, a_log, dt_bias, shape, scale):
    """The same step as torch ops (any device, fp32): (h_out, o [N, HV V])."""
    n, hk, hv, dk, dv = shape
    assert dv % 32 == 0 and hv % hk == 0, "V is whole tiles, HV a multiple of HK"
    decay, beta = gates(a, b, a_log, dt_bias)
    rep = hv // hk
    qf = _f(q).reshape(n, hk, dk).repeat_interleave(rep, 1)
    kf = _f(k).reshape(n, hk, dk).repeat_interleave(rep, 1)
    qf = qf / (qf.pow(2).sum(-1, keepdim=True) + EPS).sqrt() * scale
    kf = kf / (kf.pow(2).sum(-1, keepdim=True) + EPS).sqrt()
    s = h.reshape(n, hv, dv, dk) * decay[:, :, None, None]
    kv = (s * kf[:, :, None, :]).sum(-1)
    u = (_f(v).reshape(n, hv, dv) - kv) * beta[..., None]
    s = s + u[..., None] * kf[:, :, None, :]
    o = (s * qf[:, :, None, :]).sum(-1)
    return s.reshape(n * hv * dv, dk), o.reshape(n, hv * dv)


def inputs(shape, dev=None, seed=0):
    """A state and one step's model tensors (q, k, v bf16; a, b bf16; A_log and
    dt_bias fp32), on `dev` when given."""
    n, hk, hv, dk, dv = shape
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n * hv * dv, dk, generator=g) * 0.05
    q = torch.randn(n, hk * dk, generator=g).bfloat16()
    k = torch.randn(n, hk * dk, generator=g).bfloat16()
    v = torch.randn(n, hv * dv, generator=g).bfloat16()
    a = torch.randn(n, hv, generator=g).bfloat16()
    b = torch.randn(n, hv, generator=g).bfloat16()
    a_log = torch.rand(hv, generator=g) * 2 - 1
    dt_bias = torch.randn(hv, generator=g) * 0.5
    out = (h, q, k, v, a, b, a_log, dt_bias)
    return out if dev is None else tuple(x.to(dev) for x in out)


def relative(got, want):
    """Max error relative to the largest magnitude of `want`."""
    return ((got.cpu().double() - want).abs().max() / want.abs().max()).item()


def compare(shape, steps=1):
    """Errors against float64 after `steps` steps (the state carried) of the tile
    kernel and of the eager ops on the device, and their times per step (us)."""
    dev = kohakutt.device(0)
    scale = shape[3] ** -0.5
    host = inputs(shape)
    tt = inputs(shape, dev)
    want_h, got = host[0].double(), {"tile": tt[0], "eager": tt[0]}
    fns = {"tile": step_tile, "eager": step_eager}
    for _ in range(steps):
        args = [x.double() for x in host[1:]]
        want_h, want_o = step_eager(want_h, *args, shape, scale)
        outs = {}
        for name, fn in fns.items():
            outs[name] = fn(got[name], *tt[1:], shape, scale)
            got[name] = outs[name][0]
    out = {}
    for name in fns:
        out[f"{name} o error"] = relative(outs[name][1], want_o)
        out[f"{name} h error"] = relative(got[name], want_h)
    kernel_args = (tt[0], *layouts(*tt[1:4], *gates(*tt[4:8]), *shape[1:]), scale)
    statics = {"rows": shape[4] // 32, "group": shape[2] // shape[1]}
    out["tile kernel us"] = timed(lambda: gdn_decode(*kernel_args, **statics)) * 1e6
    for name, fn in fns.items():
        out[f"{name} step us"] = timed(lambda fn=fn: fn(*tt, shape, scale)) * 1e6
    return out


def main():
    """Runs the step at one shape (16 key heads, 48 value heads, 128 x 128 state by
    default); prints errors, times and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=int, default=1)
    parser.add_argument("--heads", type=int, nargs=2, default=(16, 48))
    parser.add_argument("--dims", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    shape = (args.seqs, *args.heads, *args.dims)
    try:
        for key, value in compare(shape, args.steps).items():
            print(f"  {key}: {value:.4g}")
        n, hk, hv, dk, dv = shape
        h = kd.like((n * hv * dv, dk), "float32")
        qk = kd.like((n * hk * 32, dk), "bfloat16")
        col = kd.like((n * hv * dv, 1), "float32")
        plan = gdn_decode.plan(
            h, qk, qk, col, col, col, dk**-0.5, rows=dv // 32, group=hv // hk
        )
        print(plan.report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
