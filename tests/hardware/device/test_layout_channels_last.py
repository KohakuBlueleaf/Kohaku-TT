"""channels_last tt tensors (storage in NHWC order): strides and
is_contiguous(memory_format) as PyTorch reports them on CPU for the same program, host
<-> device copies, views, in-place writes and the trace's contiguous clone; ops on them
give the values of the same ops on the NCHW tensor. W and C multiples of 32 read through
the Nhwc rows (one conversion); other shapes through the permute view."""

import pytest
import torch
import torch.nn.functional as F

CL = torch.channels_last
# (shape, read as Nhwc rows): W and C multiples of 32, or not.
SHAPES = [((2, 64, 8, 32), True), ((1, 32, 5, 64), True), ((2, 48, 6, 10), False)]
IDS = ["rows", "rows-n1", "permuted"]


def sample(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).bfloat16()


def conversions():
    return torch.ops.ktt._debug_layout_conversions()


def same_layout(got, want):
    """got (tt) reports want's (CPU) sizes, strides and memory-format contiguity."""
    assert got.shape == want.shape
    assert got.stride() == want.stride()
    for fmt in (torch.contiguous_format, CL):
        assert got.is_contiguous(memory_format=fmt) == want.is_contiguous(
            memory_format=fmt
        )


@pytest.mark.hw
@pytest.mark.parametrize("shape, rows", SHAPES, ids=IDS)
def test_creation_and_copies(native, shape, rows):
    x = sample(shape, 0)
    xc = x.contiguous(memory_format=CL)
    got = xc.to("tt")
    same_layout(got, xc)
    host = got.cpu()
    same_layout(host, xc)
    assert torch.equal(host, xc)
    back = got.contiguous()
    same_layout(back, x)
    assert torch.equal(back.cpu(), x)
    e = torch.empty_like(got)
    same_layout(e, torch.empty_like(xc))
    # Copies into an existing channels_last tensor, from either host layout.
    for src in (x, xc):
        e.copy_(src)
        same_layout(e, xc)
        assert torch.equal(e.cpu(), xc)
    # A host destination of either layout.
    for dst in (torch.empty_like(x), torch.empty_like(xc)):
        dst.copy_(e)
        assert torch.equal(dst, x)


@pytest.mark.hw
@pytest.mark.xfail(strict=True, reason="memory_format is ignored on tt: NCHW strides")
@pytest.mark.parametrize(
    "make",
    [
        lambda t: t.to(memory_format=CL),
        lambda t: t.contiguous(memory_format=CL),
        lambda t: t.clone(memory_format=CL),
        lambda t: torch.empty(t.shape, dtype=t.dtype, device="tt", memory_format=CL),
    ],
    ids=["to", "contiguous", "clone", "empty"],
)
def test_memory_format_argument(native, make):
    """Known gap: these give NCHW-contiguous tt tensors (only an upload of a
    channels_last CPU tensor or empty_like of a channels_last tt tensor is one)."""
    x = sample((2, 64, 8, 32), 8)
    same_layout(make(x.to("tt")), x.contiguous(memory_format=CL))


@pytest.mark.hw
@pytest.mark.parametrize("shape, rows", SHAPES, ids=IDS)
def test_reads_use_rows_where_aligned(native, shape, rows):
    """A whole channels_last read goes through one Nhwc -> Nchw conversion when W and
    C are multiples of 32, through the permute view (no conversion) otherwise."""
    x = sample(shape, 1)
    t = x.contiguous(memory_format=CL).to("tt")
    before = conversions()
    got = (t + 0).cpu()
    assert conversions() - before == (1 if rows else 0)
    assert torch.equal(got, x)


@pytest.mark.hw
@pytest.mark.parametrize("shape, rows", SHAPES, ids=IDS)
def test_ops_match_the_nchw_run(native, shape, rows):
    """Ops still read through the Nchw form: the same values as on the NCHW tensor."""
    x = sample(shape, 2)
    w = sample((16, shape[1], 3, 3), 3) * 0.05
    a, b = x.to("tt"), x.contiguous(memory_format=CL).to("tt")
    wt = w.to("tt")
    ops = {
        "add": lambda t: t + 1,
        "silu": F.silu,
        "sum": lambda t: t.sum(dim=(2, 3)),
        "conv": lambda t: F.conv2d(t, wt, padding=1),
        "group_norm": lambda t: F.group_norm(t, 16),
    }
    for name, op in ops.items():
        assert torch.equal(op(b).cpu(), op(a).cpu()), name


@pytest.mark.hw
@pytest.mark.parametrize("shape, rows", SHAPES, ids=IDS)
def test_views_resolve_through_the_storage(native, shape, rows):
    x = sample(shape, 4)
    xc = x.contiguous(memory_format=CL)
    t = xc.to("tt")
    views = {
        "channel slice": lambda v: v[:, :3],
        "width slice": lambda v: v[..., 1:5],
        "image": lambda v: v[1:] if v.shape[0] > 1 else v[:, 2:],
        "permute to NHWC": lambda v: v.permute(0, 2, 3, 1),
        "permute to NHWC, contiguous": lambda v: v.permute(0, 2, 3, 1).contiguous(),
        "tokens": lambda v: v.permute(0, 2, 3, 1).reshape(v.shape[0], -1, v.shape[1]),
        "select": lambda v: v[:, 5],
    }
    for name, view in views.items():
        got, want = view(t), view(xc)
        assert got.stride() == want.stride(), name
        assert torch.equal(got.cpu(), want), name


@pytest.mark.hw
@pytest.mark.parametrize("shape, rows", SHAPES, ids=IDS)
def test_in_place_writes(native, shape, rows):
    """Values as the same writes on the NCHW tt tensor; strides as CPU's."""
    x, y = sample(shape, 5), sample(shape, 6)
    t, ref = x.contiguous(memory_format=CL).to("tt"), x.to("tt")
    cpu = x.contiguous(memory_format=CL)
    writes = [
        lambda v, other: v.add_(1),
        lambda v, other: v.mul_(other),
        lambda v, other: v[:, :2].fill_(0.5),
        lambda v, other: v[..., 1:3].copy_(other[..., :2]),
        lambda v, other: v.copy_(other),
        lambda v, other: v.fill_(2.0),
    ]
    for i, write in enumerate(writes):
        write(t, y.to("tt"))
        write(ref, y.to("tt"))
        write(cpu, y)
        same_layout(t, cpu)
        assert torch.equal(t.cpu(), ref.cpu()), i


@pytest.mark.hw
@pytest.mark.parametrize("shape, rows", SHAPES, ids=IDS)
def test_trace_clone_is_contiguous(native, shape, rows):
    """kohakutt.trace's stable input buffers: clone(memory_format=contiguous_format)
    of a channels_last input is an NCHW-contiguous tensor of the same values."""
    x = sample(shape, 7)
    t = x.contiguous(memory_format=CL).to("tt")
    c = t.clone(memory_format=torch.contiguous_format)
    assert c.is_contiguous()
    same_layout(c, x)
    assert torch.equal(c.cpu(), x)
