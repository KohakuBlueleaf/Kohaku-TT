"""Plain tinygrad on a Tenstorrent chip: tensors, math, indexing, views, gradients. Only
the device differs from CPU code. Usage: python examples/tinygrad/01_tensors.py
"""

import kohakutt.tinygrad  # noqa: F401  registers the device "KTT" (chip 0)
from tinygrad import Tensor

dev = "KTT"

# Tensors move to and from the device like any other tinygrad device.
x = Tensor.randn(4, 8).to(dev)
w = Tensor.randn(8, 3).to(dev)
print(x.device, x.dtype, x.shape)

# Math, broadcasting, reductions and matmul run on the chip.
y = (x @ w + 1.0).relu()
print("row sums:", y.sum(axis=1).numpy())
print("softmax:", y.softmax(-1)[0].numpy())

# Slicing, padding, concatenation and views.
print("first column:", x[:, 0].numpy())
print("padded:", x.pad(((1, 1), (2, 0))).shape, "cat:", x.cat(x, dim=0).shape)
print("transpose:", x.T.shape, "reshape:", x.reshape(2, 16).shape)
print("cumsum:", x.cumsum(-1)[0].numpy())

# Gradients are computed on the device too.
a = Tensor.randn(16, 16).to(dev)
loss = (a.sin() * a).mean()
(grad,) = loss.gradient(a)
print("loss:", loss.item(), "grad on", grad.device)

# Results match CPU up to bf16 / fp32 rounding.
cpu = Tensor.randn(64, 64)
diff = abs(cpu.to(dev).tanh().numpy() - cpu.tanh().numpy()).max()
print(f"tanh vs CPU max |diff|: {diff:.2e}")
