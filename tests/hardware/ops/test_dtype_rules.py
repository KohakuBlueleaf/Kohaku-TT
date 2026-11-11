"""Regression: result dtype and value of reductions and arithmetic on bool and integer
tensors equal PyTorch's on CPU; uint8 / int8 / int16 are stored as int32 and wrap."""

import io

import pytest
import torch

G = torch.Generator().manual_seed(0)
BL = torch.rand(1000, generator=G) > 0.3
BL2 = torch.rand(1000, generator=G) > 0.6
I32 = torch.randint(-50, 50, (1000,), generator=G, dtype=torch.int32)
I64 = torch.randint(-50, 50, (1000,), generator=G)
SMALL = torch.randint(1, 3, (8,), generator=G)
F32 = torch.randn(1000, generator=G)
M2 = torch.rand(40, 37, generator=G) > 0.5
I2 = torch.randint(-9, 9, (40, 37), generator=G)
U8 = I64.abs().to(torch.uint8)
LARGE = torch.randint(0, 1000, (300, 300), generator=G)

CASES = {
    "bool.sum": lambda t: t(BL).sum(),
    "bool.sum.dim": lambda t: t(M2).sum(1),
    "bool.sum.keepdim": lambda t: t(M2).sum(0, keepdim=True),
    "bool.sum.dtype_f32": lambda t: t(BL).sum(dtype=torch.float32),
    "bool.sum.dtype_i32": lambda t: t(BL).sum(dtype=torch.int32),
    "bool.sum.0d": lambda t: t(torch.tensor(True)).sum(),
    "bool.sum.empty": lambda t: t(torch.zeros(0, dtype=torch.bool)).sum(),
    "bool.sum.large": lambda t: (t(torch.ones(300, 300)) > 0).sum(),
    "bool.sum.dim.large": lambda t: (t(torch.ones(4, 70000)) > 0).sum(1),
    "eq.sum": lambda t: (t(I64) == t(I64 % 2)).sum(),
    "i32.sum": lambda t: t(I32).sum(),
    "i64.sum": lambda t: t(I64).sum(),
    "i64.sum.dim": lambda t: t(I2).sum(0),
    "i64.sum.large": lambda t: t(LARGE).sum(),
    "i64.sum.dim.large": lambda t: t(LARGE).sum(0),
    "i64.nansum": lambda t: torch.nansum(t(I64)),
    "bool.mean.dtype": lambda t: t(BL).mean(dtype=torch.float32),
    "bool.float.mean": lambda t: t(BL).float().mean(),
    "bool.prod": lambda t: t(BL[:8]).prod(),
    "i64.prod": lambda t: t(SMALL).prod(),
    "i64.prod.dim": lambda t: t(I2[:, :3]).prod(1),
    "i32.prod.dtype": lambda t: t(SMALL.int()).prod(dtype=torch.int32),
    "bool.cumsum": lambda t: t(BL).cumsum(0),
    "i32.cumsum": lambda t: t(I32).cumsum(0),
    "bool.cumprod": lambda t: t(BL[:8]).cumprod(0),
    "i64.cumprod": lambda t: t(SMALL).cumprod(0),
    "bool.any": lambda t: t(BL).any(),
    "i64.all.dim": lambda t: t(I2).all(1),
    "bool.count_nonzero": lambda t: torch.count_nonzero(t(BL)),
    "i64.count_nonzero.dim": lambda t: torch.count_nonzero(t(I2), 1),
    "bool+bool": lambda t: t(BL) + t(BL2),
    "bool*bool": lambda t: t(BL) * t(BL2),
    "bool+i64": lambda t: t(BL) + t(I64),
    "bool+1": lambda t: t(BL) + 1,
    "bool*2.5": lambda t: t(BL) * 2.5,
    "i32+i64": lambda t: t(I32) + t(I64),
    "i64*i64": lambda t: t(I64) * t(I64).flip(0),
    "i64/i64": lambda t: t(I64) / t(I64).abs().clamp(min=1),
    "i64==i64": lambda t: t(I64) == t(I64).flip(0),
    "i64.argmax": lambda t: t(I64).argmax(),
    "i64.argmax.dim": lambda t: t(I2).argmax(1),
    "i64.max": lambda t: t(I64).max(),
    "bool.max": lambda t: t(BL).max(),
    "where.i64.f32": lambda t: torch.where(t(BL), t(I64), t(F32)),
    "where.bool.bool": lambda t: torch.where(t(BL), t(BL2), t(BL)),
    "where.i64.scalar": lambda t: torch.where(t(BL), t(I64), 0),
    "bool.uint8": lambda t: t(BL).to(torch.uint8),
    "u8.roundtrip": lambda t: t(U8),
    "i8.roundtrip": lambda t: t(I64.to(torch.int8)),
    "i16.roundtrip": lambda t: t((I64 * 300).to(torch.int16)),
    "u8.sum": lambda t: t(BL.to(torch.uint8)).sum(),
    "i8.sum": lambda t: t(I64.to(torch.int8)).sum(),
    "i16.sum": lambda t: t(I64.to(torch.int16)).sum(),
    "u8+u8": lambda t: t(U8) + t(U8),
    "u8.float": lambda t: t(U8).float(),
    "u8.view.clone": lambda t: t(U8)[3:9].clone(),
    "u8<20": lambda t: t(U8) < 20,
    "u8.argmax": lambda t: t(U8).argmax(),
    "u8.cumsum": lambda t: t(U8).cumsum(0),
    "u8.mask": lambda t: t(F32)[t((F32 > 0).to(torch.uint8)).bool()],
    "u8.wrap.mul": lambda t: t(torch.full((4,), 200, dtype=torch.uint8)) * 2,
    "u8.wrap.neg": lambda t: -t(torch.tensor([1, 2, 250], dtype=torch.uint8)),
    "u8.wrap.sum_dtype": lambda t: t(torch.full((300,), 3, dtype=torch.uint8)).sum(
        dtype=torch.uint8
    ),
    "i8.wrap.add": lambda t: t(torch.full((4,), 100, dtype=torch.int8)) + 100,
    "i8.wrap.mul": lambda t: t(I64.to(torch.int8)) * t(I64.to(torch.int8)),
    "i16.wrap.mul": lambda t: t((I64 * 300).to(torch.int16)) * 200,
    "i32.to.u8": lambda t: t(torch.tensor([300, -1, 255, 7], dtype=torch.int32)).to(
        torch.uint8
    ),
}


@pytest.mark.hw
@pytest.mark.parametrize("name", list(CASES))
def test_dtype_rule(native, name):
    want = CASES[name](lambda x: x)
    got = CASES[name](lambda x: x.to("tt")).cpu()
    assert got.dtype == want.dtype, f"{name}: {got.dtype} != {want.dtype}"
    if want.is_floating_point():
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)
    else:
        assert torch.equal(got, want), f"{name}: {got} != {want}"


@pytest.mark.hw
def test_eq_sum_item(native):
    a, b = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 3, 0])
    got = (a.to("tt") == b.to("tt")).sum().item()
    assert type(got) is int and got == 2


@pytest.mark.hw
def test_mean_of_integers_raises(native):
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        I64.to("tt").mean()


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8, torch.int8])
def test_save_load(native, dtype):
    x = torch.arange(-3, 9).to(dtype)
    buf = io.BytesIO()
    torch.save(x.to("tt"), buf)
    buf.seek(0)
    got = torch.load(buf, map_location="cpu")
    assert got.dtype == dtype and torch.equal(got, x)
