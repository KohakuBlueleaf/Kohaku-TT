"""Matmul kernels of the tile level without a card: the accumulator form, its
lowering to a verified metal-level program and its plan report."""

import pytest

import kohakutt.dsl as kd
import kohakutt.dsl.metal as km
from kohakutt.dsl.metal.body import SfpuCall, walk


def _linear(chain=None, nt=4, cols=None):
    """A linear kernel storing chain(acc, b) (b a bias row; None: acc)."""

    def fn(a: kd.Tensor, w: kd.Tensor, b: kd.Tensor, out: kd.Out):
        g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
        for i, j in g.units():
            acc = kd.accumulator((1, nt))
            hi = (j + 1) * nt if cols is None else cols(j)
            for k in kd.range(kd.num_tiles(a, -1)):
                acc += kd.load(a.tiles[i, k]) @ kd.load(w.tiles[k, j * nt : hi])
            bias = kd.load(b.tiles[0, j * nt : (j + 1) * nt]).row()
            value = acc if chain is None else chain(acc, bias)
            kd.store(out.tiles[i, j * nt : (j + 1) * nt], value)

    return kd.kernel(fn, name=f"linear_{id(fn)}")


ARGS = (kd.like((1, 512, 768)), kd.like((768, 1024)), kd.like((1024,)))


def _calls(program):
    return [
        s for s, _ in walk(program.kernels["compute"].body) if isinstance(s, SfpuCall)
    ]


def test_plain_linear_has_no_vector_work():
    program = _linear().program(*ARGS)
    km.verify(program)
    assert _calls(program) == []
    assert [r.name for r in program.inputs] == ["a", "w"]


def test_bias_and_activation_run_on_the_accumulator():
    k = _linear(lambda acc, b: kd.silu(acc + b), nt=2)
    program = k.program(*ARGS)
    km.verify(program)
    (call,) = _calls(program)
    assert call.b == 2  # the bias sits past the two accumulator slots
    source = km.source(program, "compute")
    assert "unary_bcast<BroadcastType::ROW>" in source and "matmul_block(" in source


def test_report_names_the_multiply_work():
    report = _linear(lambda acc, b: kd.silu(acc + b), nt=2).plan(*ARGS).report()
    assert "matmul form" in report and "GFLOP" in report and "binds" in report


def test_a_wide_accumulator_leaves_no_slot_for_the_bias():
    with pytest.raises(kd.PlanError, match="register slots"):
        _linear(lambda acc, b: acc + b, nt=4).plan(*ARGS)


def test_the_weight_block_follows_the_accumulator():
    with pytest.raises(kd.PlanError, match="right operand"):
        _linear(cols=lambda j: j * 4 + 2).plan(*ARGS)
