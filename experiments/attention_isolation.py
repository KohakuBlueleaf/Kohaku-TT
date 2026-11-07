"""Attention isolation: SDPA over B rows x H heads of identical inputs; each (batch,
head) pair against the pair computed alone; the worst pair's error statistics per
layout and length. KTT_SDPA_MAX_UNITS=100000 runs TT-NN's SDPA unsplit."""

import argparse

import torch
import torch.nn.functional as F

import kohakutt
from kohakutt.core.stats import diff_stats, format_stats


def main():
    """Runs each (batch, heads) layout at each length on device 0."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--lengths", type=int, nargs="*", default=[279, 1043, 1600, 1615, 4110, 5695]
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=["1x24", "1x48", "1x64", "1x128", "2x24", "3x24"],
        help="BxH",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    dev = kohakutt.device(0)
    gen = torch.Generator().manual_seed(0)
    for s in args.lengths:
        base = [
            torch.randn(1, 1, s, args.head_dim, generator=gen).bfloat16()
            for _ in range(3)
        ]
        with torch.no_grad():
            alone = F.scaled_dot_product_attention(*(t.to(dev) for t in base)).cpu()
            for layout in args.layouts:
                b, h = map(int, layout.split("x"))
                out = F.scaled_dot_product_attention(
                    *(t.expand(b, h, -1, -1).contiguous().to(dev) for t in base)
                ).cpu()
                worst = max(
                    (
                        diff_stats(out[i, j], alone[0, 0])
                        for i in range(b)
                        for j in range(h)
                    ),
                    key=lambda d: d["abs_max"],
                )
                print(f"S {s:<5} B {b} H {h:<3} {format_stats(worst)}", flush=True)


if __name__ == "__main__":
    main()
