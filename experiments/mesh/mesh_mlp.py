"""Transformer MLP blocks on a 1x2 mesh (column-parallel up, row-parallel down, bias,
residual): collectives per block and per pair of blocks, output placement, and error
statistics against float64 next to the 1-card and CPU same-dtype results.

    python experiments/mesh/mesh_mlp.py
"""

import torch

import kohakutt


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = torch.nn.Linear(dim, hidden)
        self.down = torch.nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


def blocks(n, dim, hidden, dtype):
    torch.manual_seed(0)
    return torch.nn.Sequential(*[MLP(dim, hidden) for _ in range(n)]).to(dtype)


def main():
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for dtype in (torch.bfloat16, torch.float32):
            for n in (1, 2):
                cpu = blocks(n, 256, 1024, dtype)
                x = torch.randn(64, 256, generator=torch.Generator().manual_seed(1))
                x = x.to(dtype)
                ref = blocks(n, 256, 1024, torch.float64)(x.double())
                one = blocks(n, 256, 1024, dtype).to("tt")
                one_out = one(x.to("tt")).cpu()
                model = blocks(n, 256, 1024, dtype).to("tt")
                for layer in model:
                    for p, dim in (
                        (layer.up.weight, 0),
                        (layer.up.bias, 0),
                        (layer.down.weight, 1),
                    ):
                        p.data = kohakutt.shard(p.data, dim)
                xd = x.to("tt")
                torch.tt.reset_mesh_stats()
                with torch.no_grad():
                    out = model(xd)
                placement = kohakutt.placement(out)
                stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
                fallbacks = torch.tt.mesh_fallback_ops()
                got = out.cpu()
                print(
                    f"{n} block(s) {dtype}: output {placement}; {stats}; "
                    f"fallbacks {fallbacks}"
                )
                for label, t in (("mesh", got), ("1-card", one_out), ("cpu", cpu(x))):
                    s = kohakutt.format_stats(
                        kohakutt.diff_stats(t.detach().double(), ref.detach())
                    )
                    print(f"    {label:6s} {s}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
