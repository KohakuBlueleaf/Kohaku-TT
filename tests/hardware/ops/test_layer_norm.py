"""bf16 layer norm on the route in effect: output, statistics and gradients against CPU float64,
and dx per statistics row group. Over one dim the route is ktt_norm by default and TT-NN's
moreh layer norm with KTT_CUSTOM_KERNELS=0 (run both: moreh's own last-dim statistics are wrong
past the first row of every 32 and misread for odd rows); over two dims it is moreh."""

import pytest
import torch

# (shape, normalized dims)
CASES = [
    ((1, 4096), 1),
    ((32, 1024), 1),
    ((77, 256), 1),
    ((33, 256), 1),
    ((4, 77, 256), 1),
    ((4, 64, 256), 1),
    ((4, 33, 256), 1),
    ((2, 3, 40, 256), 1),
    ((3, 40, 96), 2),
]

# Every quantity is stored in bf16. Measured 2026-11-09: y <= 1.2e-2, the rest <= 8.5e-3. The
# moreh statistics failures were 5e-2 (dx of odd batches) to 1e7.
BOUND = 2e-2


def _rel(a, ref):
    return ((a.cpu().double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("shape,nd", CASES)
def test_layer_norm_matches_float64(native, shape, nd):
    norm = shape[-nd:]
    g = torch.Generator().manual_seed(sum(shape))
    x = (torch.randn(*shape, generator=g) * 2 + 1).to(torch.bfloat16)
    dy = torch.randn(*shape, generator=g).to(torch.bfloat16)
    w = (torch.randn(*norm, generator=g) * 0.5 + 1).to(torch.bfloat16)
    b = (torch.randn(*norm, generator=g) * 0.1).to(torch.bfloat16)
    mask = [True, True, True]
    ref = torch.native_layer_norm(x.double(), norm, w.double(), b.double(), 1e-5)
    ref_grads = torch.ops.aten.native_layer_norm_backward(
        dy.double(), x.double(), norm, ref[1], ref[2], w.double(), b.double(), mask
    )
    y, mean, rstd = torch.native_layer_norm(x.to("tt"), norm, w.to("tt"), b.to("tt"), 1e-5)
    dx, dw, db = torch.ops.aten.native_layer_norm_backward(
        dy.to("tt"), x.to("tt"), norm, mean, rstd, w.to("tt"), b.to("tt"), mask
    )
    assert mean.shape == ref[1].shape and rstd.shape == ref[2].shape
    errs = {
        "y": _rel(y, ref[0]),
        "mean": _rel(mean, ref[1]),
        "rstd": _rel(rstd, ref[2]),
        "dx": _rel(dx, ref_grads[0]),
        "dweight": _rel(dw, ref_grads[1]),
        "dbias": _rel(db, ref_grads[2]),
    }
    group = shape[len(shape) - nd - 1 :]
    dx_groups = dx.cpu().double().reshape(-1, *group)
    ref_groups = ref_grads[0].reshape(-1, *group)
    for i in range(dx_groups.shape[0]):
        errs[f"dx[{i}]"] = _rel(dx_groups[i], ref_groups[i])
    assert max(errs.values()) <= BOUND, errs
