"""The library functions, comparisons and the fused multiply-add of generated
vector-unit functions (`km.sfpu_fn`, tests/km_fused.py) on the card, against the tile
ops of the same name and against torch."""

import pytest
import torch

import kohakutt.dsl.metal as km
from km_chain import run
from km_fused import FLASH_A, cases, checks, fused_program, on_pack

SHAPE = (256, 512)


def _tensors(count, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(SHAPE, generator=g) * 2 for _ in range(count)]


@pytest.mark.hw
@pytest.mark.parametrize("case", ["gate", "exp", "tanh", "sqrt", "rsqrt"])
def test_library_function_equals_its_tile_op(native, case):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    inputs, ops, function = cases()[case]
    chain = km.Launcher(fused_program(machine, f"{case}_ops", inputs, ops))
    fused = km.Launcher(fused_program(machine, f"{case}_fn", inputs, function))
    tensors = [t.to(dev) for t in _tensors(inputs, 6)]
    assert torch.equal(run(fused, tensors).cpu(), run(chain, tensors).cpu()), case


@pytest.mark.hw
def test_recip_is_the_refined_estimate(native):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    _, _, function = cases()["recip"]
    fused = km.Launcher(fused_program(machine, "recip_fn", 1, function))
    (x,) = _tensors(1, 7)
    got = run(fused, [x.to(dev)]).cpu()
    assert ((got - 1 / x).abs() <= 2.0**-22 * (1 / x).abs()).all()


@pytest.mark.hw
def test_select_is_exact_and_a_multiply_add_rounds_once(native):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    data = _tensors(4, 8)
    out = {}
    for name, (inputs, function, reference) in checks().items():
        fused = km.Launcher(fused_program(machine, f"{name}_fn", inputs, function))
        got = run(fused, [t.to(dev) for t in data[:inputs]]).cpu()
        out[name] = (got, reference(*(t.double() for t in data[:inputs])))
    got, want = out["select"]
    assert torch.equal(got, want.float())
    a, b, c = (t.double() for t in data[:3])
    got, want = out["mad"]
    error = (got.double() - want).abs()
    assert (error <= 2.0**-23 * ((a * b).abs() + c.abs())).all()
    # Closer to the exact value than the product rounded to fp32 and then added.
    separate = ((data[0] * data[1] + data[2]).double() - want).abs()
    assert error.mean() < separate.mean()
    got, want = out["clamp"]
    assert ((got.double() - want).abs() <= 2.0**-23 * ((a * b).abs() + c.abs())).all()
    assert got.abs().max() <= 1.5


@pytest.mark.hw
def test_flash_exp_equals_ktt_flash_bit_for_bit(native):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    inputs, hand, function = cases()["flash_exp"]
    raw = km.Launcher(fused_program(machine, "flash_exp_raw", inputs, hand))
    fused = km.Launcher(fused_program(machine, "flash_exp_fn", inputs, function))
    g = torch.Generator().manual_seed(11)
    # Scores x and per-row b = 639 - A m (m the row max), as ktt_flash's state.
    x = torch.randn(SHAPE, generator=g) * 40
    m = x.max(dim=1, keepdim=True).values
    b = (639 - FLASH_A * m).expand(SHAPE).contiguous()
    x[0, :4] = torch.tensor([-1e30, -1e4, m[0, 0] - 900, m[0, 0]])
    tensors = [x.to(dev), b.to(dev)]
    want = run(raw, tensors).cpu()
    got = run(fused, tensors).cpu()
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    assert got[0, :3].eq(0).all() and (got > 0).sum() > 0.9 * got.numel()
    exact = torch.exp2(x.double() * FLASH_A + b.double() - 639)
    live = exact > 1e-30
    error = ((got.double() - exact).abs() / exact)[live]
    assert error.max() < 1.8e-3, error.max()


@pytest.mark.hw
def test_flash_exp_on_the_pack_thread_equals_the_math_thread(native):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    _, _, function = cases()["flash_exp"]
    math = km.Launcher(fused_program(machine, "flash_exp_fn", 2, function))
    pack = km.Launcher(fused_program(machine, "flash_exp_pack", 2, on_pack()))
    g = torch.Generator().manual_seed(12)
    x = torch.randn(SHAPE, generator=g) * 40
    m = x.max(dim=1, keepdim=True).values
    b = (639 - FLASH_A * m).expand(SHAPE).contiguous()
    tensors = [x.to(dev), b.to(dev)]
    want = run(math, tensors).cpu()
    for _ in range(3):
        got = run(pack, tensors).cpu()
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.hw
def test_bf16_registers_round_once_at_the_store(native):
    dev = native.device(0)
    machine = km.Machine(*native.compute_grid())
    inputs, ops, function = cases()["swiglu"]
    kw = {"dtype": "bfloat16", "fp32": False}
    chain = km.Launcher(fused_program(machine, "swiglu_bf16_ops", inputs, ops, **kw))
    fused = km.Launcher(
        fused_program(machine, "swiglu_bf16_fn", inputs, function, **kw)
    )
    data = [t.bfloat16() for t in _tensors(inputs, 9)]
    tensors = [t.to(dev) for t in data]
    got = run(fused, tensors).cpu().float()
    want = run(chain, tensors).cpu().float()
    exact = torch.nn.functional.silu(data[0].double()) * data[1].double()
    assert torch.isfinite(got).all()
    # One bf16 rounding of the fp32 result; the chain of ops rounds after each op.
    assert ((got.double() - exact).abs() <= 2.0**-6 * exact.abs() + 1e-30).all()
    assert ((got - want).abs() <= 2.0**-7 * want.abs() + 1e-30).all()
