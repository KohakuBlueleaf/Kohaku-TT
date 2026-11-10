"""Kohaku-TT's softmax over the last dim (ktt_softmax): error against float64 next to
CPU at the same dtype for widths 1 to 4096 and ranks 1 to 4, -inf masks, an all -inf
row, large values, inf in the tile padding, gradients, no compile for a new shape and
bit-identical repeats."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

DTYPES = [torch.bfloat16, torch.float32]
WIDTHS = [1, 31, 32, 33, 70, 197, 1000, 4096]
SHAPES = [
    (5,),
    (1000,),
    (33, 70),
    (64, 197),
    (2, 40, 33),
    (3, 4, 31),
    (2, 3, 33, 70),
    (8, 2, 32, 1000),
]


@pytest.fixture(autouse=True)
def _own_route(monkeypatch):
    """Every softmax of these tests on ktt_softmax, whatever the size rule picks."""
    monkeypatch.setenv("KTT_OWN_SOFTMAX", "1")


def _rel(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


def _check(x, scale=1.0):
    """softmax of x on tt against float64, beside CPU at x's dtype; returns tt's."""
    ref = torch.softmax(x.double(), -1)
    cpu = _rel(torch.softmax(x, -1), ref)
    got = torch.softmax(x.to("tt"), -1).cpu()
    assert got.dtype == x.dtype
    err = _rel(got, ref)
    # One rounding of an fp32 result: the error sits at CPU's (fp32: within 2^-22).
    floor = 2e-6 if x.dtype == torch.float32 else 0.0
    assert err <= 1.5 * cpu + floor, (tuple(x.shape), err, cpu)
    return got


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
def test_widths_match_float64(native, dtype, width):
    gen = torch.Generator().manual_seed(width)
    _check((torch.randn(37, width, generator=gen) * 4).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_ranks_match_float64(native, dtype, shape):
    gen = torch.Generator().manual_seed(sum(shape))
    _check((torch.randn(*shape, generator=gen) * 3).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(64, 197), (3, 1000), (2, 4096)], ids=str)
def test_neg_inf_mask(native, dtype, shape):
    """-inf entries give exactly 0 and leave the rest of the row exact; an all -inf
    row gives NaN (bf16: non-finite, see limitations)."""
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(*shape, generator=gen).to(dtype)
    keep = torch.rand(*shape, generator=gen) < 0.6
    keep[:, 0] = True
    x = x.masked_fill(~keep, float("-inf"))
    x[1] = float("-inf")
    got = torch.softmax(x.to("tt"), -1).cpu()
    assert not torch.isfinite(got[1]).any()
    rows = [0] + list(range(2, shape[0]))
    ref = torch.softmax(x[rows].double(), -1)
    cpu = _rel(torch.softmax(x[rows], -1), ref)
    floor = 2e-6 if dtype == torch.float32 else 0.0
    assert _rel(got[rows], ref) <= 1.5 * cpu + floor
    assert torch.all(got[rows][~keep[rows]] == 0)


@pytest.mark.parametrize("width", [70, 4096])
def test_non_finite_rows(native, width):
    """fp32 rows holding +inf or NaN give NaN rows (as torch); the other rows stay
    exact."""
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(40, width, generator=gen)
    x[1, 5] = float("inf")
    x[3, width - 1] = float("nan")
    x[5, 2] = -float("nan")
    assert x[5, 2].view(torch.int32) < 0
    got = torch.softmax(x.to("tt"), -1).cpu()
    assert torch.isnan(got[[1, 3, 5]]).all()
    rows = [0, 2, 4] + list(range(6, 40))
    ref = torch.softmax(x[rows].double(), -1)
    assert _rel(got[rows], ref) <= 1.5 * _rel(torch.softmax(x[rows], -1), ref) + 2e-6


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [70, 1000, 4096])
def test_large_values(native, dtype, width):
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(40, width, generator=gen) * 1e4
    x[:, 3] = 1e4
    x[5] = -1e4
    _check(x.to(dtype))


def _dirty(x):
    """x on tt whose tile padding (last two dims) holds inf."""
    *lead, r, c = x.shape
    big = torch.full(
        (*lead, (r + 31) // 32 * 32 + 32, (c + 31) // 32 * 32 + 32),
        float("inf"),
        dtype=x.dtype,
    )
    big[..., :r, :c] = x
    return big.to("tt")[..., :r, :c]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [(33, 70), (4, 33, 197), (2, 1000), (3, 40, 4096)], ids=str
)
def test_padding_inf_bit_identical(native, dtype, shape):
    """inf in the tile padding of a ragged input: the result equals the clean run."""
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(*shape, generator=gen).to(dtype)
    clean = torch.softmax(x.to("tt"), -1).cpu()
    dirty = torch.softmax(_dirty(x), -1).cpu()
    assert torch.equal(clean, dirty)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gradients(native, dtype):
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(4, 33, 197, generator=gen).to(dtype)
    dy = torch.randn(4, 33, 197, generator=gen).to(dtype)
    xd = x.double().requires_grad_(True)
    torch.softmax(xd, -1).backward(dy.double())
    xc = x.clone().requires_grad_(True)
    torch.softmax(xc, -1).backward(dy)
    xt = x.to("tt").requires_grad_(True)
    F.softmax(xt, -1).backward(dy.to("tt"))
    cpu = _rel(xc.grad, xd.grad)
    floor = 2e-6 if dtype == torch.float32 else 0.0
    assert _rel(xt.grad.cpu(), xd.grad) <= 3 * cpu + floor


def test_new_shape_compiles_nothing(native):
    for shape in ((40, 300), (3, 50, 4100), (1, 777)):
        x = torch.randn(*shape)
        torch.softmax(x.to("tt"), -1).cpu()
        before = kohakutt.counters()["kernel_compiles"]
    for shape in ((41, 301), (2, 65, 2000), (1, 900)):
        x = torch.randn(*shape)
        torch.softmax(x.to("tt"), -1).cpu()
    assert kohakutt.counters()["kernel_compiles"] == before


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_bit_identical(native, dtype):
    gen = torch.Generator().manual_seed(9)
    xt = torch.randn(8, 197, 197, generator=gen).to(dtype).to("tt")
    first = torch.softmax(xt, -1).cpu()
    for _ in range(3):
        assert torch.equal(torch.softmax(xt, -1).cpu(), first)
