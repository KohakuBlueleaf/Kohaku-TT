"""A model larger than one card on a 1x2 mesh with automatic placement: a stack of bf16
Linear layers (8192 x 8192, GELU between) whose parameters exceed one chip's DRAM,
created on the mesh and run forward. Prints parameter bytes, per-chip and mesh memory
(torch.tt.mem_get_info, kohakutt.memory_stats per chip), placements, forward time, and
the output's error statistics against float64 on the CPU for the first layers' chain
recomputed on the host, next to CPU bf16.

    python experiments/mesh/mesh_large.py --layers 320
"""

import argparse
import time

import torch

import kohakutt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=320)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--check", type=int, default=4, help="layers checked on CPU")
    args = parser.parse_args()
    dev = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        free, total = torch.tt.mem_get_info()
        print(f"mesh DRAM free {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB")
        layers = []
        g = torch.Generator().manual_seed(0)
        host = []
        for i in range(args.layers):
            w = torch.randn(args.dim, args.dim, generator=g, dtype=torch.bfloat16)
            w = w / args.dim**0.5
            if i < args.check:
                host.append(w)
            layers.append(w.to(dev))
        param_bytes = sum(w.numel() * w.element_size() for w in layers)
        free, total = torch.tt.mem_get_info()
        print(
            f"{args.layers} weights {args.dim}x{args.dim} bf16 = "
            f"{param_bytes / 2**30:.1f} "
            f"GiB; placement {kohakutt.placement(layers[0])}; mesh free after "
            f"{free / 2**30:.1f} GiB; per chip "
            f"{kohakutt.memory_stats(dev, per_chip=True)}"
        )
        x = torch.randn(args.rows, args.dim, generator=g, dtype=torch.bfloat16)

        def forward(n):
            y = x.to(dev)
            for w in layers[:n]:
                y = torch.nn.functional.gelu(y @ w.t())
            return y

        with torch.no_grad():
            forward(args.layers)
            torch.tt.synchronize()
            start = time.perf_counter()
            forward(args.layers)
            torch.tt.synchronize()
            ms = (time.perf_counter() - start) * 1e3
            print(f"forward through {args.layers} layers: {ms:.1f} ms (warm)")
            got = forward(args.check).cpu().double()
        ref = x.double()
        base = x
        for w in host:
            ref = torch.nn.functional.gelu(ref @ w.double().t())
            base = torch.nn.functional.gelu(base @ w.t())
        s = kohakutt.format_stats(kohakutt.diff_stats(got, ref))
        print(f"first {args.check} layers: device {s}")
        s = kohakutt.format_stats(kohakutt.diff_stats(base.double(), ref))
        print(f"first {args.check} layers: cpu    {s}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
