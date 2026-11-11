"""One online-softmax step written at the tile level: for a block of scores s and the
running row max m and row sum (the denominator) den, the new max, the block's
probabilities p and the new sum, in one launch reading each row of s once.
"""

import math

import torch

import kohakutt.dsl as kd


@kd.kernel
def softmax_step(
    s: kd.Tensor,
    m: kd.Tensor,
    den: kd.Tensor,
    p: kd.Out,
    m_out: kd.Out,
    den_out: kd.Out,
    scale: kd.Runtime[float],
):
    """s [..., R, N] (N a multiple of 32), m and den [..., R, 1] fp32."""
    for r in kd.grid(kd.num_tile_rows(s)):
        row = kd.load(s.tiles[r, :])
        m_old = kd.load(m.tiles[r])
        den_old = kd.load(den.tiles[r])
        m_new = kd.maximum(m_old, kd.reduce_max(row))
        e = kd.exp((row - m_new.col()) * scale)
        alpha = kd.exp((m_old - m_new) * scale)
        kd.store(p.tiles[r, :], e)
        kd.store(m_out.tiles[r], m_new)
        kd.store(den_out.tiles[r], den_old * alpha + kd.reduce_sum(e, exact=True))


def step_eager(s, m, den, scale):
    """The same step with torch ops (any device): (p, m_new, den_new), p in s's
    dtype."""
    sf = s.float()
    m_new = torch.maximum(m, sf.amax(-1, keepdim=True))
    e = torch.exp((sf - m_new) * scale)
    alpha = torch.exp((m - m_new) * scale)
    return e.to(s.dtype), m_new, den * alpha + e.sum(-1, keepdim=True)


def step_reference(s, m, den, scale):
    """The step in float64 on the CPU: (p, m_new, den_new)."""
    s, m, den = (t.cpu().double() for t in (s, m, den))
    m_new = torch.maximum(m, s.amax(-1, keepdim=True))
    e = torch.exp((s - m_new) * scale)
    return e, m_new, den * torch.exp((m - m_new) * scale) + e.sum(-1, keepdim=True)


def inputs(rows, cols, dev, heads=1, seed=0):
    """Scores [1, heads, rows, cols] bf16 and a previous state (m, den) fp32."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, heads, rows, cols, generator=g) * 4
    m = torch.randn(1, heads, rows, 1, generator=g) * 4 + 6
    den = torch.rand(1, heads, rows, 1, generator=g) * 50 + 1
    return s.bfloat16().to(dev), m.to(dev), den.to(dev)


def compare(rows, cols, heads=1, scale=0.125):
    """Errors against the float64 step, of the tile kernel and of the eager torch ops
    on the device: m (absolute), p (in half bf16 ulps), den (relative)."""
    import kohakutt

    dev = kohakutt.device(0)
    s, m, den = inputs(rows, cols, dev, heads)
    want = step_reference(s, m, den, scale)
    out = {}
    for name, got in (
        ("tile", softmax_step(s, m, den, scale)),
        ("eager", step_eager(s, m, den, scale)),
    ):
        p, m_new, den_new = (t.cpu().double() for t in got)
        out[f"{name} m error"] = (m_new - want[1]).abs().max().item()
        ulp = torch.finfo(torch.bfloat16).eps * want[0].abs() / 2
        out[f"{name} p error (half ulps)"] = ((p - want[0]).abs() / ulp).max().item()
        error = ((den_new - want[2]) / want[2]).abs().max().item()
        out[f"{name} den error"] = error
    q = softmax_check(s, scale).cpu().double()
    ref = torch.softmax(s.cpu().double() * scale, -1)
    error = ((q - ref) / ref).abs().max().item()
    out["softmax from the empty state error (relative)"] = error
    return out


def softmax_check(s, scale):
    """The step from the empty state (m = -inf, den = 0): p / den_new is
    softmax(scale * s)."""
    shape = (*s.shape[:-1], 1)
    m = torch.full(shape, -math.inf, dtype=torch.float32).to(s.device)
    den = torch.zeros(shape, dtype=torch.float32).to(s.device)
    p, _, den_new = softmax_step(s, m, den, scale)
    return p.float() / den_new


def main():
    """Runs the step at one shape; prints its errors, the eager ops' and the plan."""
    import argparse

    import kohakutt

    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--keys", type=int, default=1024)
    args = parser.parse_args()
    try:
        for key, value in compare(args.rows, args.keys, args.heads).items():
            print(f"  {key}: {value:.3g}")
        s = kd.like((1, args.heads, args.rows, args.keys))
        m = kd.like((1, args.heads, args.rows, 1), "float32")
        print(softmax_step.plan(s, m, m, 0.125).report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
