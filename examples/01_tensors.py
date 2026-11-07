"""Plain PyTorch on a Tenstorrent chip: tensors, math, indexing, views, autograd. Only
the device differs from CPU / CUDA code. Usage: python examples/01_tensors.py
"""

import torch

import kohakutt

dev = kohakutt.device(0)  # chip 0 as a torch.device ("tt:0")

# Tensors move to and from the device like any other torch device.
x = torch.randn(4, 8, device=dev)
w = torch.randn(8, 3).to(dev)
print(x.device, x.dtype, x.shape)

# Math, broadcasting, reductions and matmul run on the chip.
y = torch.relu(x @ w + 1.0)
print("row sums:", y.sum(dim=1).cpu())
print("softmax:", torch.softmax(y, dim=-1)[0].cpu())

# Indexing, slicing, views and in-place updates.
x[:, 0] = 0.0
top = x.topk(2, dim=1).values
print("first column zeroed:", x[:, 0].cpu(), "top-2:", tuple(top.shape))
print(
    "transpose:",
    tuple(x.t().contiguous().shape),
    "reshape:",
    tuple(x.reshape(2, 16).shape),
)

# Autograd: backward runs on the device too.
a = torch.randn(16, 16, device=dev, requires_grad=True)
loss = (a.sin() * a).mean()
loss.backward()
print("loss:", loss.item(), "grad on", a.grad.device)

# Results match CPU up to bf16 / fp32 rounding.
cpu = torch.randn(64, 64)
diff = (torch.tanh(cpu.to(dev)).cpu() - torch.tanh(cpu)).abs().max().item()
print(f"tanh vs CPU max |diff|: {diff:.2e}")
