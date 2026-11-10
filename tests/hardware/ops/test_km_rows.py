"""Control words, a counter on another core and endless loops on the card: the copy
program of tests/km_rows.py returns its input, with rows dealt out in turn and with
rows taken from the counter, at fewer and at more tile rows than cores."""

import pytest
import torch

from km_rows import rows_program, runtime

SHAPES = [(256, 96), (2, 1000, 64), (8192, 64)]


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rows_from_words_and_a_counter_copy_the_input(native, dtype):
    import kohakutt.dsl.metal as km

    dev = native.device(0)
    cols, rows = native.compute_grid()
    name = "float32" if dtype == torch.float32 else "bfloat16"
    launcher = km.Launcher(rows_program(km.Machine(cols, rows), name))
    for seed, shape in enumerate(SHAPES):
        x = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
        x = x.to(dtype)
        for dynamic in (0, 1):
            values = runtime(shape, cols * rows, dynamic)
            (out,) = launcher(
                {"x": x.to(dev)}, outputs={"out": (list(shape), dtype)}, runtime=values
            )
            assert torch.equal(out.cpu(), x), (shape, dynamic)
