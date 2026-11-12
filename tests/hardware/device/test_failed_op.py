"""After a tt op fails on the stream, its error is raised once, and a tensor whose op
failed or was skipped raises when used instead of being read."""

import contextlib

import pytest
import torch


def _too_big():
    # A fill too large for device memory: its allocation fails on the stream (the
    # call itself returns, no error is pending before it).
    return torch.full((24, 2**30), 1.0, dtype=torch.bfloat16, device="tt")


@pytest.mark.hw
def test_tensor_of_a_failed_op_raises_when_used(native):
    x = torch.randn(24, 64).to("tt")
    big = _too_big()
    with pytest.raises(RuntimeError):
        torch.tt.synchronize()
    for use in (lambda: big.cpu(), lambda: (big[:, :64] + x).cpu()):
        with pytest.raises(RuntimeError, match="never computed"):
            use()
    assert torch.equal((x + 1).cpu(), x.cpu() + 1)


@pytest.mark.hw
def test_capture_whose_prepare_pass_caught_an_op_error_raises(native):
    def caught(t):
        big = _too_big()
        with contextlib.suppress(RuntimeError):
            torch.tt.synchronize()
        return big[:, :64] * 3 + t

    x = torch.randn(24, 64).to("tt")
    with pytest.raises(RuntimeError):
        native.capture(caught, x, warmup=1)
    for _ in range(2):
        assert torch.equal((x + 1).cpu(), x.cpu() + 1)
