"""Tile-level kernels over runs of tiles (GroupNorm) without a card: plan, verified
lowering, report, refusals."""

import pytest

import kohakutt.dsl as kd
import kohakutt.dsl.metal as km


@kd.kernel
def gn(
    x: kd.Tensor,
    w: kd.Tensor,
    b: kd.Tensor,
    out: kd.Out,
    eps: kd.Runtime[float],
    groups: kd.Static[int] = 32,
):
    per = kd.num_tile_rows(x) // kd.size(x, 1) // kd.size(x, 0) * kd.num_tiles(x, -1)
    cpg = kd.size(x, 1) // groups
    run = cpg * per
    n = kd.size(x, 2) * kd.size(x, 3) * cpg
    for u in kd.grid(kd.size(x, 0) * groups):
        blk = kd.load(x.tiles[u * run : (u + 1) * run]).cast(kd.f32)
        g0 = u % groups * cpg
        wc = kd.load(w.tiles[g0 : g0 + cpg]).each(per)
        bc = kd.load(b.tiles[g0 : g0 + cpg]).each(per)
        mean = kd.reduce_sum(blk, exact=True) / n
        d = blk - mean.all()
        r = kd.rsqrt(kd.reduce_sum(d * d, exact=True) / n + eps)
        kd.store(
            out.tiles[u * run : (u + 1) * run], (d * r.all() * wc + bc).cast(out.dtype)
        )


def _args(shape=(2, 320, 64, 64)):
    return (
        kd.like(shape),
        kd.like((shape[1], 32, 32)),
        kd.like((shape[1], 32, 32)),
        1e-5,
    )


def test_a_group_is_a_run_with_two_exact_sums():
    plan = gn.plan(*_args())
    assert plan.design.form == "flat" and plan.units == 64
    assert [v.kind for v in plan.design.values] == ["exact", "chain", "exact", "chain"]
    program = gn.program(*_args())
    km.verify(program)
    assert "per-channel tiles" in plan.report()


def test_per_channel_runs_take_their_own_buffers():
    # 256 tiles per group (VAE); the 16 channels' weight and bias tiles apart.
    plan = gn.plan(*_args((1, 512, 128, 128)))
    program = gn.program(*_args((1, 512, 128, 128)))
    seg = [n for n in program.sized_values if n != "seg"]
    assert len(seg) == 2 and plan.units == 32


def test_a_padded_height_over_one_tile_is_refused():
    with pytest.raises(kd.PlanError, match="heights <= 32 or whole tiles"):
        gn.plan(*_args((2, 64, 48, 64)))
