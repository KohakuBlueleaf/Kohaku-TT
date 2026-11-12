"""The capture flag the optimizer check reads is per thread: inside a capture on one
thread, a non-capturable optimizer's step raises there and runs on another thread."""

import threading

import pytest
import torch

import kohakutt
from kohakutt.torch_backend import trace

pytestmark = pytest.mark.skipif(
    not kohakutt.AVAILABLE, reason="needs the native build (the optimizer check)"
)


def _step():
    p = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([p], lr=0.1)
    p.grad = torch.ones(4)
    opt.step()


def test_flag_is_per_thread():
    trace._local.depth = 1
    try:
        assert trace.capturing()
        with pytest.raises(RuntimeError, match=r"capturable=True"):
            _step()
        seen, errors = [], []

        def other():
            seen.append(trace.capturing())
            try:
                _step()
            except RuntimeError as error:
                errors.append(error)

        worker = threading.Thread(target=other)
        worker.start()
        worker.join()
        assert seen == [False] and errors == []
    finally:
        trace._local.depth = 0
    _step()
