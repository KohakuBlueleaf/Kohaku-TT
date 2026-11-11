"""Tile-level kernels with carried values (flash attention) without a card: plan,
verified lowering, report and refusals."""

import pytest

import kohakutt.dsl as kd
import kohakutt.dsl.metal as km


def _flash(causal_default=0, transposed=True):
    @kd.kernel(name=f"flash_t{causal_default}{int(transposed)}")
    def flash(
        q: kd.Tensor,
        k: kd.Tensor,
        v: kd.Tensor,
        masks: kd.Tensor,
        out: kd.Out,
        scale: kd.Runtime[float],
        causal: kd.Static[int] = causal_default,
    ):
        qt, kt = kd.num_tiles(q, -2) // kd.size(q, 0), kd.num_tiles(k, -2) // kd.size(
            k, 0
        )
        blocks = kd.cdiv(qt, 4)
        for u in kd.grid(kd.size(q, 0) * blocks):
            head, b = u // blocks, u % blocks
            first = head * qt + b * 4
            n = kd.minimum(4, qt - b * 4)
            qb = kd.load(q.tiles[first : first + n, :])
            m, den, acc = kd.carry(-1e30), kd.carry(0.0), kd.carry(0.0)
            seen = kd.minimum(kt, b * 4 + n) if causal else kt
            for j in kd.range(kd.cdiv(seen, 4)):
                k0 = head * kt + j * 4
                nk = kd.minimum(4, seen - j * 4)
                kb = kd.load(k.tiles[k0 : k0 + nk, :])
                vb = kd.load(v.tiles[k0 : k0 + nk, :])
                raw = qb @ (kb.T if transposed else kb)
                s = kd.mask_scores(
                    raw * scale, masks, b * 4, j * 4, kd.size(k, -2), causal
                )
                m_new = kd.maximum(m, kd.reduce_max(s))
                p = kd.exp(s - m_new.col())
                alpha = kd.exp(m - m_new)
                den.next(den * alpha + kd.reduce_sum(p))
                acc.next(acc * alpha.col() + p @ vb)
                m.next(m_new)
            kd.store(out.tiles[first : first + n, :], (acc / den.col()).cast(out.dtype))

    return flash


def _args(bh=4, s=300, d=64):
    t = kd.like((bh, s, d), "bfloat16")
    return (t, t, t, kd.like((5, 32, 32), "float32"), 0.125)


@pytest.mark.parametrize("causal", [0, 1])
def test_flash_plans_and_verifies(causal):
    flash = _flash(causal)
    plan = flash.plan(*_args())
    assert plan.design.form == "attention" and plan.units == 4 * 3
    km.verify(flash.program(*_args()))
    assert plan.l1 <= km.L1_BUDGET
    report = plan.report()
    assert "carried   tile value" in report and "masked" in report
    assert "next" in flash.ir(*_args())


def test_scores_without_a_transposed_key_are_refused():
    with pytest.raises(kd.PlanError, match="q @ k.T and p @ v"):
        _flash(0, transposed=False).plan(*_args())


def test_a_carry_outside_the_grid_is_refused():
    @kd.kernel(name="carry_outside")
    def bad(x: kd.Tensor, out: kd.Out):
        kd.carry(0.0)
        for t in kd.grid(kd.num_tile_rows(x)):
            kd.store(out.tiles[t, :], kd.load(x.tiles[t, :]))

    with pytest.raises(kd.PlanError, match="inside the grid loop"):
        bad.plan(kd.like((64, 64), "bfloat16"))


def test_score_masks_mark_the_padding_and_the_upper_triangle():
    m = kd.score_masks(100)
    assert m.shape == (5, 32, 32)
    assert (m[0] == 0).all() and (m[4] < -1e29).all()
    assert m[1][0, 3] == 0 and m[1][0, 4] < -1e29  # 100 = 3 * 32 + 4 keys
    assert m[2][0, 1] < -1e29 and m[2][1, 0] == 0
