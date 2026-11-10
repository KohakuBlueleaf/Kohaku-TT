"""A standard training loop on a Tenstorrent chip: a small MLP learns a synthetic task
with AdamW; model, loss and optimizer code is plain tinygrad.
Usage: python examples/tinygrad/02_training.py
"""

import numpy as np

import kohakutt.tinygrad  # noqa: F401  registers the device "KTT" (chip 0)
from tinygrad import Context, Tensor, nn


class SmallMLP:
    """Flatten -> Linear -> SiLU -> Linear -> SiLU -> Linear."""

    def __init__(self, classes=4):
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, classes)

    def __call__(self, x):
        return self.head(self.l2(self.l1(x.flatten(1)).silu()).silu())


def batch(n, rng):
    """Images with a bright quadrant; the label is which quadrant."""
    labels = rng.integers(0, 4, n)
    images = rng.standard_normal((n, 1, 16, 16)).astype(np.float32) * 0.3
    for i, q in enumerate(labels):
        r, c = divmod(int(q), 2)
        images[i, 0, r * 8 : r * 8 + 8, c * 8 : c * 8 + 8] += 1.0
    return Tensor(images, device="KTT"), Tensor(labels.astype(np.int32), device="KTT")


def main(steps=40):
    """Trains on KTT and reports loss and accuracy."""
    Tensor.manual_seed(0)
    rng = np.random.default_rng(0)

    model = SmallMLP()
    for p in nn.state.get_parameters(model):
        p.to_("KTT")
    opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-3)

    for step in range(steps):
        images, labels = batch(64, rng)
        with Context(TRAINING=1):
            loss = model(images).sparse_categorical_crossentropy(labels)
            opt.zero_grad()
            loss.backward()
            value = (
                loss.item()
            )  # the loss of this step's weights: read before the optimizer updates them
            opt.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {value:.4f}")

    images, labels = batch(256, rng)
    logits = model(images)
    predicted = logits.argmax(axis=1)
    print(f"eval accuracy: {(predicted == labels).mean().item():.3f}")

    # The same weights on the CPU backend give the same logits up to rounding.
    for p in nn.state.get_parameters(model):
        p.to_("CPU")
    diff = abs(model(images.to("CPU")).numpy() - logits.numpy()).max()
    print(f"logits vs CPU max |diff|: {diff:.2e}")


if __name__ == "__main__":
    main()
