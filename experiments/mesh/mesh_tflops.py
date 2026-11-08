"""Achieved throughput on 1 card and on a 1x2 mesh (automatic placement), same shapes:
square bf16 matmuls, stacks of MLP blocks (Linear d -> 4d -> d, GELU, residual), and a
weight-streaming chain (64 rows through 8192 x 8192 Linear weights). Prints warm ms
(median of 10, after 3 untimed calls), TFLOPS counted as 2 FLOP per multiply-add of
the matmuls only, and, for the weight-streaming chain, weight bytes read per second.

    python experiments/mesh/mesh_tflops.py --cards 1
    python experiments/mesh/mesh_tflops.py --cards 2
"""

import argparse
import statistics
import time

import torch

import kohakutt


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = torch.nn.Linear(dim, 4 * dim)
        self.down = torch.nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


def timed(fn, repeat=10):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.tt.synchronize()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            torch.tt.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    dev = kohakutt.device(0) if args.cards == 1 else kohakutt.mesh([0, 1], shape=(1, 2))
    tag = f"{args.cards} card(s)"
    try:
        g = torch.Generator().manual_seed(0)
        for n in (2048, 4096, 8192):
            a = torch.randn(n, n, generator=g, dtype=torch.bfloat16).to(dev)
            b = torch.randn(n, n, generator=g, dtype=torch.bfloat16).to(dev)
            ms = timed(lambda a=a, b=b: a @ b)
            print(
                f"{tag} matmul {n}^3: {ms:.3f} ms, {2 * n**3 / ms / 1e9:.1f} TFLOPS "
                f"(a {kohakutt.placement(a)})",
                flush=True,
            )
            del a, b
        for dim, rows in ((1024, 256), (2048, 1024), (4096, 2048)):
            torch.manual_seed(0)
            model = torch.nn.Sequential(*[MLP(dim) for _ in range(4)])
            model = model.to(torch.bfloat16).to(dev)
            x = torch.randn(rows, dim, generator=g, dtype=torch.bfloat16).to(dev)
            ms = timed(lambda model=model, x=x: model(x))
            flop = 4 * 2 * (2 * rows * dim * 4 * dim)
            print(
                f"{tag} 4 MLP blocks {dim}->{4 * dim}, {rows} rows: {ms:.2f} ms, "
                f"{flop / ms / 1e9:.1f} TFLOPS",
                flush=True,
            )
            del model, x
        layers = 24 if args.cards == 1 else 48
        ws = [
            (torch.randn(8192, 8192, generator=g, dtype=torch.bfloat16) / 90.5).to(dev)
            for _ in range(layers)
        ]
        x = torch.randn(64, 8192, generator=g, dtype=torch.bfloat16).to(dev)

        def chain():
            y = x
            for w in ws:
                y = torch.nn.functional.gelu(y @ w.t())
            return y

        ms = timed(chain, repeat=5)
        flop = 2 * 64 * 8192 * 8192 * layers
        weight_bytes = 8192 * 8192 * 2 * layers
        print(
            f"{tag} weight streaming, 64 rows x {layers} weights 8192^2: {ms:.1f} ms, "
            f"{flop / ms / 1e9:.1f} TFLOPS, "
            f"weights read at {weight_bytes / ms / 1e6:.0f} GB/s",
            flush=True,
        )
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
