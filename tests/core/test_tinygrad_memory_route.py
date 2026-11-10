"""Structure detection of the memory-op read route (host side): which reads are
padded strided views."""

import numpy as np
import pytest

pytest.importorskip("tinygrad.engine")

from kohakutt.tinygrad_backend import executor
from kohakutt.tinygrad_backend.executor import Unsupported, analyse_read, decompose


def test_split_axis_is_two_pieces():
    # qkv split: buffer [T, 3, H, D] read as [3, T, H, D] via div / mod
    T, H, D = 4, 2, 3
    flat = np.arange(T * 3 * H * D).reshape(T, 3, H, D)
    ia = flat.transpose(1, 0, 2, 3).reshape(3 * T, H, D)
    pieces, off = decompose(ia)
    assert off == 0
    assert [(n, s) for _, n, s in pieces] == [
        (3, H * D),
        (T, 3 * H * D),
        (H, D),
        (D, 1),
    ]


def test_non_affine_index_is_refused():
    ia = np.array([0, 5, 1, 9, 2, 7])
    with pytest.raises(Unsupported):
        decompose(ia)


def test_padding_box_and_triangle():
    ia = np.arange(6 * 6).reshape(6, 6)
    box = np.zeros((6, 6), bool)
    box[1:5, 2:6] = True
    axes, counts, starts, steps, full, pieces, off = analyse_read(
        ia, box, [6, 6], 36, ["a", "b"]
    )
    assert (counts, starts, steps, full, off) == ([4, 4], [1, 2], [1, 1], [6, 6], 8)
    tri = np.tril(np.ones((6, 6), bool))
    with pytest.raises(Unsupported):
        analyse_read(ia, tri, [6, 6], 36, ["a", "b"])


def test_zero_interleave_gate_is_a_step():
    ia = np.arange(8)[None, :].repeat(3, 0) // 2 + 4 * np.arange(3)[:, None]
    valid = np.zeros((3, 8), bool)
    valid[:, 1::2] = True  # every other element, offset 1
    axes, counts, starts, steps, full, pieces, off = analyse_read(
        ia, valid, [3, 8], 12, ["a", "b"]
    )
    assert (counts, starts, steps, full) == ([3, 4], [0, 1], [1, 2], [3, 8])


def test_broadcast_axis_is_dropped_and_zero_piece_kept():
    ia = np.broadcast_to(np.arange(4)[None, :], (3, 4))
    axes, widths, lo, hi, full, pieces, off = analyse_read(
        ia, None, [3, 4], 4, ["a", "b"]
    )
    assert axes == ("b",) and widths == [4]
    ia = np.tile(np.arange(4), 3)  # one axis of 12 = (3 x stride 0, 4 x stride 1)
    axes, widths, lo, hi, full, pieces, off = analyse_read(ia, None, [12], 4, ["a"])
    assert [(n, s) for _, n, s in pieces] == [(3, 0), (4, 1)]


def test_module_exports_counters():
    assert hasattr(executor, "ROUTES") and hasattr(executor, "ROUTE_SKIPS")
