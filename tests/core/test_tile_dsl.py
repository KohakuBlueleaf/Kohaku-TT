"""The tile level (`kohakutt.dsl`) without a card: trace into the tile IR, the plan,
the lowering to metal-level programs (verified and emitted) and the plan report."""

import pytest

import kohakutt.dsl as kd
import kohakutt.dsl.metal as km

f = km.sfpu


@kd.kernel
def swiglu(x: kd.Tensor, g: kd.Tensor, out: kd.Out, block: kd.Static[int] = 4):
    for t in kd.grid(kd.num_tiles(out), block=block):
        a = kd.load(x.tiles[t])
        b = kd.load(g.tiles[t])
        kd.store(out.tiles[t], a * kd.sigmoid(a) * b)


@kd.kernel
def norm(x: kd.Tensor, scale: kd.Tensor, out: kd.Out, eps: kd.Runtime[float]):
    g = kd.resident(scale.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        r = kd.rsqrt(ms + eps)
        kd.store(out.tiles[t, :], (row * r.col() * g.row()).cast(out.dtype))


@kd.kernel
def half(x: kd.Tensor, cos: kd.Tensor, sin: kd.Tensor, out: kd.Out):
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :]).cast(kd.f32)
        p = t % kd.num_tile_rows(cos)
        c, s = kd.load(cos.tiles[p, :]), kd.load(sin.tiles[p, :])
        h = kd.size(x, -1) // 2
        rotated = kd.cat([-row[..., h:], row[..., :h]])
        kd.store(out.tiles[t, :], (row * c + rotated * s).cast(out.dtype))


@kd.kernel
def pairs(x: kd.Tensor, table: kd.Tensor, out: kd.Out):
    for t in kd.grid(kd.num_tiles(out), block=2):
        a = kd.load(x.tiles[t]).cast(kd.f32)
        b = kd.load(table.tiles[t % kd.num_tiles(table)])
        re = a.even * b.even - a.odd * b.odd
        im = a.even * b.odd + a.odd * b.even
        kd.store(out.tiles[t], kd.pairs(re, im).cast(out.dtype))


def _norm_args(tokens=512, width=1536, dtype="bfloat16"):
    return kd.like((1, tokens, width), dtype), kd.like((width,)), 1e-6


def _half_args(tokens=576):
    return kd.like((1, 16, tokens, 64)), kd.like((tokens, 64)), kd.like((tokens, 64))


def test_trace_records_the_tile_ir():
    text = swiglu.ir(kd.like((64, 128)), kd.like((64, 128)))
    assert "grid unit < tiles(out), block 4" in text
    assert "sigmoid(%0)" in text and "store out.tiles[unit]" in text


def test_swiglu_lowers_to_one_function_equal_to_the_metal_expression():
    program = swiglu.program(kd.like((64, 128)), kd.like((64, 128)))
    km.verify(program)
    (fn,) = [h for h in program.kernels["compute"].helpers if hasattr(h, "operands")]
    a, b = f.operands(2)
    want = km.sfpu_fn(fn.name, f.silu(a) * b)
    assert fn.code == want.code and fn.init == "silu"
    assert sorted(km.emit(program)) == ["compute", "dm0", "dm1"]


def test_a_cast_inside_a_chain_rounds_there():
    @kd.kernel
    def rounded(x: kd.Tensor, y: kd.Tensor, out: kd.Out):
        for t in kd.grid(kd.num_tiles(out)):
            a = kd.load(x.tiles[t]).cast(kd.f32)
            kd.store(out.tiles[t], (a * a).cast(kd.bf16) + kd.load(y.tiles[t]))

    args = (kd.like((64, 64), "float32"), kd.like((64, 64), "float32"))
    (fn,) = [h for h in rounded.program(*args).kernels["compute"].helpers]
    assert fn.code.count("float32_to_bf16_rne") >= 1
    body = fn.code.split("float32_to_bf16_rne")[0]
    assert "sfpi::dst_reg[b * 32]" in fn.code and "t0 * t0" in body


def test_registers_follow_the_operands():
    bf = swiglu.program(kd.like((64, 128)), kd.like((64, 128)))
    assert bf.kernels["compute"].options["fp32_dest"] is False
    assert norm.program(*_norm_args()).kernels["compute"].options["fp32_dest"]
    wide = kd.kernel(swiglu.fn, name="swiglu_wide", dest="fp32")
    program = wide.program(kd.like((64, 128)), kd.like((64, 128)))
    assert program.kernels["compute"].options["fp32_dest"] is True


def test_norm_is_two_passes_over_a_held_row():
    plan = norm.plan(*_norm_args())
    assert plan.design.form == "row" and plan.mode == "hold"
    held = [op for op in plan.design.operands if op.held]
    assert [op.tensor.name for op in held] == ["x"]
    source = norm.source(*_norm_args(), role="compute")
    assert "two_pass_stats_update_rows<true>(0, 0, " in source
    assert "transpose_dest<DST_ACCUM_MODE>(2);" in source
    km.verify(norm.program(*_norm_args()))


def test_norm_report_has_the_cost_line():
    report = norm.plan(*_norm_args()).report()
    assert "moved" in report and "launches 1" in report
    assert "binds" in report and "row held in L1" in report


def test_a_new_size_in_its_class_reuses_the_program():
    first = norm.program(*_norm_args(512))
    for tokens in (520, 640, 1000):
        assert norm.program(*_norm_args(tokens)) is first
    assert norm.program(*_norm_args(512, 1280)) is first


def test_a_row_too_wide_for_l1_is_read_twice():
    plan = norm.plan(*_norm_args(64, 8192, "float32"))
    assert plan.mode == "reread"
    assert "read twice" in plan.report()


def test_rotate_half_branches_on_the_cat_piece():
    program = half.program(*_half_args())
    km.verify(program)
    source = km.source(program, "compute")
    assert "if (j < " in source and "} else {" in source
    fns = [h for h in program.kernels["compute"].helpers if hasattr(h, "operands")]
    assert len(fns) == 2 and all(fn.operands == 4 for fn in fns)


def test_rotate_half_refuses_a_ragged_half():
    args = (kd.like((1, 2, 64, 96)), kd.like((64, 96)), kd.like((64, 96)))
    with pytest.raises(kd.PlanError, match="multiple of 32"):
        half.plan(*args)


def test_complex_pairs_lower_to_a_pair_function():
    program = pairs.program(kd.like((1, 4, 197, 64)), kd.like((197, 64), "float32"))
    km.verify(program)
    (fn,) = [h for h in program.kernels["compute"].helpers if hasattr(h, "operands")]
    a, b = f.operands(2)
    want = km.sfpu_fn(
        fn.name,
        {
            "even": a.even * b.even - a.odd * b.odd,
            "odd": a.even * b.odd + a.odd * b.even,
        },
    )
    assert fn.code == want.code


def test_errors_name_the_kernel_line():
    @kd.kernel
    def bad(x: kd.Tensor, out: kd.Out):
        for t in kd.grid(kd.num_tile_rows(x)):
            row = kd.load(x.tiles[t, :])
            s = kd.reduce_sum(row * row, exact=True)
            kd.store(out.tiles[t, :], row * s.col())

    with pytest.raises(kd.PlanError, match="pad declared.*test_tile_dsl.py"):
        bad.plan(kd.like((64, 64)))


def test_a_runtime_size_is_no_python_condition():
    @kd.kernel
    def bad(x: kd.Tensor, out: kd.Out):
        if kd.size(x, -1) > 0:
            pass

    with pytest.raises(kd.PlanError, match="truth value"):
        bad.plan(kd.like((64, 64)))
