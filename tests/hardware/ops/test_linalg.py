"""Regression: Kohaku-TT's LU / Cholesky / QR linear algebra, the fp32 product, and
scatter amax with duplicate indices agree with PyTorch on CPU."""

import pytest
import torch


def _close(got, want):
    torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-5)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.hw
def test_lu_factor_det_inv_solve(native):
    a = 0.3 * torch.randn(3, 9, 9) + torch.eye(9)
    b = torch.randn(3, 9, 4)
    lu, piv = torch.linalg.lu_factor(a.to("tt"))
    want_lu, want_piv = torch.linalg.lu_factor(a)
    assert torch.equal(piv.cpu(), want_piv)
    _close(lu, want_lu)
    _close(torch.linalg.det(a.to("tt")), torch.linalg.det(a))
    _close(torch.linalg.inv(a.to("tt")), torch.linalg.inv(a))
    _close(torch.linalg.solve(a.to("tt"), b.to("tt")), torch.linalg.solve(a, b))


@pytest.mark.hw
def test_cholesky_qr_lstsq(native):
    x = torch.randn(2, 7, 5)
    spd = x.transpose(1, 2) @ x + torch.eye(5)
    _close(torch.linalg.cholesky(spd.to("tt")), torch.linalg.cholesky(spd))
    q, r = torch.linalg.qr(x.to("tt"))
    want_q, want_r = torch.linalg.qr(x)
    _close(q, want_q)
    _close(r, want_r)
    b = torch.randn(2, 7, 3)
    _close(
        torch.linalg.lstsq(x.to("tt"), b.to("tt")).solution,
        torch.linalg.lstsq(x, b).solution,
    )


@pytest.mark.hw
def test_prod_fp32(native):
    x = 1 + 0.1 * torch.randn(4, 300)
    _close(x.to("tt").prod(-1), x.prod(-1))


@pytest.mark.hw
def test_scatter_amax_duplicates(native):
    x = torch.randn(5, 5)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 4, 4, 3, 0])
    src = torch.randn(10, 5)
    got = x.to("tt").index_reduce(0, idx.to("tt"), src.to("tt"), "amax").cpu()
    assert torch.equal(got, x.index_reduce(0, idx, src, "amax"))
