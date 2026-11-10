"""A small transformer block under TinyJit on a Tenstorrent chip: the first calls
trace the kernels, later calls replay them. Usage: python examples/tinygrad/03_jit.py
"""

import numpy as np

import kohakutt.tinygrad  # noqa: F401  registers the device "KTT" (chip 0)
from tinygrad import Tensor, TinyJit, nn


class Block:
    """Pre-norm transformer block: causal self-attention and an MLP, with residuals."""

    def __init__(self, dim=128, heads=4):
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def __call__(self, x):
        b, t, d = x.shape
        qkv = (
            self.qkv(self.ln1(x))
            .reshape(b, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attn = qkv[0].scaled_dot_product_attention(qkv[1], qkv[2], is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(self.fc1(self.ln2(x)).gelu())


def main():
    """Runs the block on CPU, then jitted on KTT; the outputs agree call after call."""
    Tensor.manual_seed(0)
    block = Block()
    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal((2, 32, 128)).astype(np.float32) for _ in range(5)]
    expected = [block(Tensor(x)).numpy() for x in inputs]  # CPU

    for p in nn.state.get_parameters(block):
        p.to_("KTT")

    @TinyJit
    def forward(x):
        return block(x).realize()

    for i, x in enumerate(inputs):
        out = forward(Tensor(x, device="KTT")).numpy()
        phase = "trace" if i < 2 else "replay"
        rel = abs(out - expected[i]).max() / abs(expected[i]).max()
        print(
            f"call {i} ({phase}): vs CPU, largest difference / largest value {rel:.1e}"
        )


if __name__ == "__main__":
    main()
