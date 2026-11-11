"""Tile-level row kernels with column values (several row reductions, column inputs
and outputs) without a card: plan, lowering (verified), report, refusals."""

import pytest

import kohakutt.dsl as kd
import kohakutt.dsl.metal as km


@kd.kernel
def step(
    s: kd.Tensor,
    m: kd.Tensor,
    den: kd.Tensor,
    p: kd.Out,
    m_out: kd.Out,
    den_out: kd.Out,
    scale: kd.Runtime[float],
):
    for r in kd.grid(kd.num_tile_rows(s)):
        row = kd.load(s.tiles[r, :])
        m_old, den_old = kd.load(m.tiles[r]), kd.load(den.tiles[r])
        m_new = kd.maximum(m_old, kd.reduce_max(row))
        e = kd.exp((row - m_new.col()) * scale)
        kd.store(p.tiles[r, :], e)
        kd.store(m_out.tiles[r], m_new)
        alpha = kd.exp((m_old - m_new) * scale)
        kd.store(den_out.tiles[r], den_old * alpha + kd.reduce_sum(e, exact=True))


@kd.kernel
def row_max(x: kd.Tensor, out: kd.Out):
    for r in kd.grid(kd.num_tile_rows(x)):
        kd.store(out.tiles[r], kd.reduce_max(kd.load(x.tiles[r, :])))


@kd.kernel
def summed_chain(x: kd.Tensor, out: kd.Out):
    for r in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[r, :])
        e = kd.exp(row)
        a = kd.reduce_sum(e * row, exact=True)
        kd.store(out.tiles[r, :], e * kd.reciprocal(a).col())


def _args(rows=256, cols=1024):
    s = kd.like((1, 8, rows, cols), "bfloat16")
    m = kd.like((1, 8, rows, 1), "float32")
    return s, m, m, 0.125


def test_values_come_in_order_max_chain_sum():
    plan = step.plan(*_args())
    kinds = [v.kind for v in plan.design.values]
    assert kinds == ["max", "chain", "exact"]
    assert plan.design.form == "columns" and plan.design.dest == "fp32"


def test_the_program_verifies_and_reduces_on_both_units():
    program = step.program(*_args())
    km.verify(program, l1_budget=km.L1_BUDGET)
    text = km.dump(program)
    assert "ReduceInit(kind=max, dim=row" in text and "Stats(kind=sum" in text
    source = km.source(program, "compute")
    assert "reduce_tile" in source


def test_column_outputs_take_the_column_inputs_shape_and_dtype():
    variant, call, env = step._offline(_args(), {})[0]
    assert call.outputs["m_out"] == ([1, 8, 256, 1], "float32")
    assert call.outputs["p"][0] == [1, 8, 256, 1024]


def test_a_column_output_without_a_column_input_is_fp32_rows_by_one():
    _, call, _ = row_max._offline((kd.like((3, 64, 96)),), {})[0]
    assert call.outputs["out"] == ([3, 64, 1], "float32")


def test_report_lists_the_column_values():
    report = step.plan(*_args()).report()
    assert "row max on the matrix unit" in report
    assert "row sum of a stored row" in report and "column" in report


def test_a_sum_of_the_stored_row_is_added_up_by_the_packer():
    design = step.plan(*_args()).design
    (value,) = [v for v in design.values if v.kind == "exact"]
    assert design.fused == {id(value): 0}
    text = km.dump(step.program(*_args()))
    assert "PackAccumulate" in text and "Stats(kind=save" not in text


def test_a_ragged_row_sum_stays_on_the_statistics_pass():
    assert summed_chain.plan(kd.like((2, 64, 320))).design.fused == {}


def test_a_sum_of_a_computed_row_is_exact():
    plan = summed_chain.plan(kd.like((2, 64, 320)))
    (value,) = plan.design.values[:1]
    assert value.kind == "exact"


def test_a_ragged_row_is_refused_by_the_matrix_unit_max():
    with pytest.raises(kd.PlanError, match="multiple of 32"):
        step.plan(*_args(64, 100))


def test_a_new_size_in_its_class_reuses_the_program():
    a = step.program(*_args(256, 1024))
    b = step.program(*_args(64, 960))
    assert a is b
