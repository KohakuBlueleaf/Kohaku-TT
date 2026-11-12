"""Small helpers shared by the scripts in this directory: seeded inputs, a bf16 cast
that keeps complex buffers, the relative error against a CPU reference, and a
synchronized timer.
"""

import time

import torch

import kohakutt


def random_image(res, batch=1, seed=0):
    """Seeded pixel values in [-1, 1], [batch, 3, res, res] fp32 (exact in bf16)."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, 3, res, res, generator=gen) * 2 - 1).bfloat16().float()


def random_tokens(tokens, batch=1, vocab=20000, seed=0):
    """Seeded token ids [batch, tokens]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, tokens), generator=gen)


def to_bf16(model):
    """Casts `model`'s floating parameters and buffers to bfloat16 and leaves complex
    ones (rotary tables) as they are, shared and registered as before; a plain
    .to(torch.bfloat16) keeps only their real part."""
    return model._apply(lambda t: t.to(torch.bfloat16) if t.is_floating_point() else t)


def rel_l2(y, ref):
    """Relative L2 error of `y` against `ref`, both moved to the CPU in fp64."""
    y, ref = y.detach().cpu().double(), ref.detach().cpu().double()
    return ((y - ref).norm() / ref.norm()).item()


def timed(fn, dev, repeats=3):
    """(last output, first-call seconds, median ms of `repeats` more calls); every
    call is synchronized, so the times include the device work."""
    times = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        out = fn()
        kohakutt.synchronize(dev)
        times.append(time.perf_counter() - start)
    return out, times[0], sorted(times[1:])[repeats // 2] * 1e3


def report(name, first, median_ms, err=None, against="CPU"):
    """One result line: first-call time, median time and the error if given."""
    line = f"{name:9s} first call {first:6.2f} s, median {median_ms:8.1f} ms"
    print(line if err is None else f"{line}, rel L2 vs {against} {err:.2e}")
