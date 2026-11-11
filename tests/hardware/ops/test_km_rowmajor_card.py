"""Row-major data on the card (tests/km_rowmajor.py): a tensor read by rows, tilized,
copied out as tiles and untilized into a tensor written by rows, both equal to the
input; new row counts compile nothing."""

import pytest
import torch

from km_rowmajor import WIDTH, rows_program, run


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rows_tilize_and_untilize(native, dtype):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    launcher = km.Launcher(rows_program(km.Machine(cols, rows), name))
    for seed, blocks in enumerate([1, 5, 130]):
        x = torch.randn(
            blocks * 32, WIDTH, generator=torch.Generator().manual_seed(seed)
        )
        x = x.to(dtype)
        native.synchronize()
        before = native.counters()["kernel_compiles"]
        out, y = (t.cpu() for t in run(launcher, x.to(dev)))
        assert seed == 0 or native.counters()["kernel_compiles"] == before
        assert torch.equal(out, x), blocks
        assert torch.equal(y, x), blocks
