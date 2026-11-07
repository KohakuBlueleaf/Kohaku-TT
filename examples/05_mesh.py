"""Two chips as one torch device: kohakutt.mesh([0, 1]) is a 1x2 mesh with tensors
replicated on both chips. Usage: python examples/05_mesh.py
"""

import torch
import torch.nn as nn

import kohakutt


def main():
    """Runs a model forward / backward on the 1x2 mesh and checks it against CPU."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 4))
    x = torch.randn(32, 128)
    expected = model(x).detach()

    mesh = kohakutt.mesh([0, 1])  # chips 0 and 1 as one torch.device
    print(mesh, "spans", kohakutt.mesh_size(mesh), "chips")
    model.to(mesh)
    out = model(x.to(mesh))
    out.sum().backward()
    diff = (out.detach().cpu() - expected).abs().max().item()
    print(f"forward vs CPU max |diff|: {diff:.2e}")
    print("grad of first layer on", model[0].weight.grad.device)


if __name__ == "__main__":
    main()
