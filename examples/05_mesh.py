"""Two chips as one torch device: kohakutt.mesh([0, 1]) is a 1x2 mesh. Tensors of 1 MiB
or more are sharded over the chips when they are created; Linear pairs are placed
column -> row by default, and kohakutt.colwise / rowwise / replicate mark a module's
own placement. Usage: python examples/05_mesh.py
"""

import torch
from torch import nn

import kohakutt


class Block(nn.Module):
    """LayerNorm, then Linear d -> 4d -> d with GELU, plus the residual."""

    def __init__(self, d):
        super().__init__()
        self.norm = kohakutt.replicate(nn.LayerNorm(d))  # small: a copy on each chip
        self.up = kohakutt.colwise(nn.Linear(d, 4 * d))  # output features split
        self.down = kohakutt.rowwise(nn.Linear(4 * d, d))  # input features split

    def forward(self, x):
        return x + self.down(nn.functional.gelu(self.up(self.norm(x))))


def main():
    """Runs two blocks on the 1x2 mesh, prints placements and collectives, replays a
    captured trace, and compares the output with CPU float64."""
    torch.manual_seed(0)
    model = nn.Sequential(Block(1024), Block(1024)).bfloat16()
    x = torch.randn(512, 1024, dtype=torch.bfloat16)
    with torch.no_grad():
        reference = model.double()(x.double())
    model.bfloat16()

    mesh = kohakutt.mesh([0, 1])
    model.to(mesh)
    for name, p in model[0].named_parameters():
        print(f"{name:12s} {tuple(p.shape)} {kohakutt.placement(p)}")

    with torch.no_grad():
        xt = x.to(mesh)
        out = model(xt)
        print("collectives:", {k: v for k, v in torch.tt.mesh_stats().items() if v})
        step = kohakutt.capture(model, xt)  # one trace for the whole forward
        replayed = step(xt)
    print("eager  ", kohakutt.format_stats(kohakutt.diff_stats(out.cpu(), reference)))
    print(
        "replay ", kohakutt.format_stats(kohakutt.diff_stats(replayed.cpu(), reference))
    )
    print("memory per chip:", kohakutt.memory_stats(mesh, per_chip=True))
    kohakutt.close()


if __name__ == "__main__":
    main()
