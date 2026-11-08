"""S(0) <-> S(1) conversions on a 1x2 mesh: values against the source bit for bit, the
conversion kinds used, and µs per conversion (median of 10 after a warm call) for
[rows, cols] bf16 / fp32 tensors. Run once as is (all-to-all) and once with
KTT_MESH_ALL_TO_ALL=0 (all-gather, then partition).

    python experiments/mesh/mesh_alltoall.py
"""

import os
import statistics
import time

import torch

import kohakutt

SHAPES = [(256, 1024), (1024, 1024), (4096, 4096), (8192, 8192), (96, 1024)]


def timed(fn, repeat=10):
    fn()
    torch.tt.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.tt.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    mode = os.environ.get("KTT_MESH_ALL_TO_ALL", "1")
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for dtype in (torch.bfloat16, torch.float32):
            for shape in SHAPES:
                x = torch.randn(shape, generator=torch.Generator().manual_seed(3))
                x = x.to(dtype)
                a = torch.ops.ktt.distribute(x.to("tt"), "R,S(0)")
                torch.tt.reset_mesh_stats()
                b = torch.ops.ktt.distribute(a, "R,S(1)")
                back = torch.ops.ktt.distribute(b, "R,S(0)")
                stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
                exact = torch.equal(b.cpu(), x) and torch.equal(back.cpu(), x)
                us = timed(lambda a=a: torch.ops.ktt.distribute(a, "R,S(1)"))
                print(
                    f"KTT_MESH_ALL_TO_ALL={mode} {dtype} {shape}: S(0) -> S(1) "
                    f"{us:.0f} µs; values exact {exact}; "
                    f"conversions (both ways) {stats}"
                )
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
