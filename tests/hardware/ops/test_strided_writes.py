"""In-place writes into views with steps (strides that skip elements) of dense storage:
copy_, fill_ and in-place arithmetic, against the same writes on CPU (exact values)."""

import pytest
import torch


def _ints(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 8, shape, generator=g).to(dtype)


def _mrope(base, src):
    # The Qwen3.5 text rotary's recomposition: base[0][..., 1:33:3] = base[1][..., ...].
    base[0][..., 1:33:3] = src[1, ..., 1:33:3]


def _every_other_column(base, src):
    base[:, ::2] = src[:, :3]


def _fill_grid(base, src):
    base[1::2, ::3].fill_(5)


def _add_transposed(base, src):
    base.t()[::2] += src.t()[::2]


def _mul_last_dim(base, src):
    base[..., ::4].mul_(2)


def _as_strided(base, src):
    base.view(-1).as_strided((4, 5), (7, 2), 3).copy_(src.view(-1)[:20].view(4, 5))


CASES = {
    "mrope": (_mrope, (3, 1, 16, 32)),
    "every-other-column": (_every_other_column, (6, 6)),
    "fill-grid": (_fill_grid, (7, 9)),
    "add-transposed": (_add_transposed, (5, 8)),
    "mul-last-dim": (_mul_last_dim, (2, 3, 16)),
    "as-strided": (_as_strided, (6, 8)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_strided_write_matches_cpu(native, case, dtype):
    fn, shape = CASES[case]
    base, src = _ints(shape, 1, dtype), _ints(shape, 2, dtype)
    want = base.clone()
    fn(want, src)
    got = base.to("tt")
    fn(got, src.to("tt"))
    assert torch.equal(got.cpu(), want)


def test_strided_copy_int32(native):
    base, src = _ints((4, 12), 3, torch.int32), _ints((4, 12), 4, torch.int32)
    want = base.clone()
    want[:, 1::3] = src[:, ::3]
    got = base.to("tt")
    got[:, 1::3] = src.to("tt")[:, ::3]
    assert torch.equal(got.cpu(), want)
