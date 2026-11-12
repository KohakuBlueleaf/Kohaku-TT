"""ktt_flash's key-lengths plan: every core of a lane takes the lane sender's batch (so
the same key length and K / V block count), also a core without rows whose first row is
clipped to its chunk's end. With the batch taken from that first row (the code before
the fix) the decode lanes disagree, which hangs the lane sender."""

import itertools

import pytest
import torch

import kohakutt

pytestmark = pytest.mark.skipif(
    not kohakutt.AVAILABLE, reason="needs the native build (torch.ops.ktt)"
)

GRIDS = [(11, 10), (13, 10)]


def plan(shape, causal=False, r=13, grid=(11, 10), old=False):
    return torch.ops.ktt._debug_flash_lengths_plan(list(shape), causal, r, *grid, old)


def test_decode_lanes_disagreed_before():
    # B 1, H 14, Sq 1: lanes longer than one core; the last head's receivers took
    # batch 1 (past the end).
    assert not plan((1, 14, 14, 1, 500), old=True).startswith("ok")
    assert plan((1, 14, 14, 1, 500)).startswith("ok")


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("causal", [False, True])
def test_every_lane_core_takes_the_senders_batch(grid, causal):
    shapes = itertools.product(
        [1, 2, 3, 32, 33], [(1, 1), (8, 2), (14, 14), (32, 8)], [1, 16, 32, 100]
    )
    for b, (h, hk), sq in shapes:
        for r in (13, 4, 1):
            got = plan((b, h, hk, sq, 700), causal, r, grid)
            assert got.startswith("ok") or got == "no plan", (b, h, hk, sq, r, got)
