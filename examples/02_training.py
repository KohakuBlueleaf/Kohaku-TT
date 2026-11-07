"""A standard training loop on a Tenstorrent chip: a small CNN learns a synthetic task
with AdamW; model, loss and optimizer code is unchanged PyTorch.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import kohakutt


class SmallCNN(nn.Module):
    """Conv -> BatchNorm -> ReLU -> pool, twice, then a linear classifier."""

    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def batch(n, gen):
    """Images with a bright quadrant; the label is which quadrant."""
    labels = torch.randint(0, 4, (n,), generator=gen)
    images = torch.randn(n, 1, 16, 16, generator=gen) * 0.3
    for i, q in enumerate(labels.tolist()):
        r, c = divmod(q, 2)
        images[i, 0, r * 8 : r * 8 + 8, c * 8 : c * 8 + 8] += 1.0
    return images, labels


def main():
    """Trains on tt and reports loss and accuracy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    model = SmallCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(args.steps):
        images, labels = batch(64, gen)
        logits = model(images.to(dev))
        loss = F.cross_entropy(logits, labels.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    model.eval()
    images, labels = batch(256, gen)
    with torch.no_grad():
        predicted = model(images.to(dev)).argmax(dim=1).cpu()
    print(f"eval accuracy: {(predicted == labels).float().mean().item():.3f}")


if __name__ == "__main__":
    main()
