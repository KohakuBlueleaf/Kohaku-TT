"""torch.optim.Adam / AdamW with fused=True on tt (one ktt_adam program per parameter)
against the same optimizer on CPU in float64, over four steps: parameters and both
moments; fp32 and bf16; weight decay, maximize; amsgrad on the composite route."""

import pytest
import torch

SHAPES = [(64, 96), (96,), (8, 4, 3, 3), (33, 70)]

# (optimizer, keyword arguments)
CASES = [
    (torch.optim.AdamW, {}),
    (torch.optim.AdamW, {"weight_decay": 0.1, "betas": (0.8, 0.95)}),
    (torch.optim.AdamW, {"maximize": True}),
    (torch.optim.Adam, {}),
    (torch.optim.Adam, {"weight_decay": 0.1}),
    (torch.optim.Adam, {"weight_decay": 0.1, "maximize": True}),
    (torch.optim.AdamW, {"amsgrad": True}),
]

# fp32: the SFPU's fp32 arithmetic against float64. bf16: every stored value is rounded
# to 8 bits per step.
BOUND = {torch.float32: 2e-5, torch.bfloat16: 3e-2}


def _run(device, dtype, make, kwargs, programs=None):
    gen = torch.Generator().manual_seed(0)
    values = [torch.randn(*s, generator=gen).to(dtype) for s in SHAPES]
    grads = [
        [torch.randn(*s, generator=gen).to(dtype) for s in SHAPES] for _ in range(4)
    ]
    ref = device == "cpu"
    params = [torch.nn.Parameter((v.double() if ref else v).to(device)) for v in values]
    opt = make(params, lr=0.05, fused=not ref, **kwargs)
    for step in grads:
        for p, g in zip(params, step):
            p.grad = (g.double() if ref else g).to(device)
        if programs is not None:
            torch.tt.synchronize()
            before = programs()
        opt.step()
        if programs is not None:
            torch.tt.synchronize()
            count = programs() - before
    state = [opt.state[p] for p in params]
    out = {
        "p": [p.detach().cpu().double() for p in params],
        "m": [s["exp_avg"].cpu().double() for s in state],
        "v": [s["exp_avg_sq"].cpu().double() for s in state],
    }
    return (out, count) if programs is not None else out


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_fused_adam_matches_float64(native, dtype, case):
    make, kwargs = CASES[case]
    want = _run("cpu", dtype, make, kwargs)
    got, programs = _run(
        "tt", dtype, make, kwargs, lambda: native.counters()["programs"]
    )
    for key in want:
        for i, (a, b) in enumerate(zip(got[key], want[key])):
            assert a.shape == b.shape
            error = ((a - b).norm() / b.norm()).item()
            assert error <= BOUND[dtype], (key, SHAPES[i], error)
    if not kwargs.get("amsgrad"):
        # One own program per parameter, plus each step counter's increment.
        assert programs <= 2 * len(SHAPES), programs
