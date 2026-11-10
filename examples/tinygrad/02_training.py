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


def gradients(model, images, labels, device):
    """The loss and the parameter gradients of one batch, computed on `device`."""
    params = nn.state.get_parameters(model)
    for p in params:
        p.grad = None
        p.to_(device)
    with Context(TRAINING=1):
        loss = model(images.to(device)).sparse_categorical_crossentropy(
            labels.to(device)
        )
        loss.backward()
    grads = [p.grad.numpy() for p in params]
    for p in params:
        p.grad = None
    return loss.item(), grads


def main(steps=40):
    """Trains on KTT and reports loss and accuracy, each checked against the CPU."""
    Tensor.manual_seed(0)
    rng = np.random.default_rng(0)

    model = SmallMLP()
    # One batch on both backends with the same weights: losses and gradients agree.
    images, labels = batch(64, np.random.default_rng(1))
    loss_cpu, grads_cpu = gradients(model, images, labels, "CPU")
    loss_ktt, grads_ktt = gradients(model, images, labels, "KTT")
    worst = max(abs(a - b).max() / abs(b).max() for a, b in zip(grads_ktt, grads_cpu))
    print(f"first batch: loss KTT {loss_ktt:.4f}, CPU {loss_cpu:.4f}")
    print(f"first batch: gradients, largest difference / largest value {worst:.1e}")

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

    # The trained weights on the CPU backend: same logits up to the matmul precision.
    for p in nn.state.get_parameters(model):
        p.to_("CPU")
    ref = model(images.to("CPU")).numpy()
    got = logits.numpy()
    rel = abs(got - ref).max() / abs(ref).max()
    agree = (got.argmax(1) == ref.argmax(1)).mean()
    print(f"logits vs CPU: largest difference / largest logit {rel:.1e}")
    print(f"predictions equal to the CPU's: {agree:.3f}")


if __name__ == "__main__":
    main()
