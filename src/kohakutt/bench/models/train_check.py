"""Experiment: eager training on tt vs CPU: MLP / block / CNN with AdamW.

Same seed, data and bf16 model on both devices; reports the per-step loss curves, the
final weights' error statistics, host transfers in the loop and CPU-fallback calls.
    python -m kohakutt.bench.models.train_check [--steps 20] [--foreach]
"""

import argparse
import json

import torch

import kohakutt
from kohakutt.core.stats import diff_stats


class Block(torch.nn.Module):
    """A causal transformer block (attention + GELU MLP, pre-norm) with a vocab head."""

    def __init__(self, dim=64, heads=4, seq=32, vocab=16):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = torch.nn.LayerNorm(dim), torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * dim, dim),
        )
        self.head = torch.nn.Linear(dim, vocab)
        self.register_buffer("causal", torch.tril(torch.ones(seq, seq)).bool())

    def forward(self, x):
        b, t, c = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=-1)
        q, k, v = (z.view(b, t, h, c // h).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * (c // h) ** -0.5
        att = att.masked_fill(~self.causal[:t, :t], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return self.head(x)


def make_cnn():
    """A small CNN: 2 x (conv, batch norm, ReLU), max / adaptive avg pool, linear."""
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def make_model(kind):
    """Seeded bf16 model: "mlp" (64 -> 128 -> 10), "block" (Block) or "cnn"."""
    torch.manual_seed(0)
    if kind == "block":
        return Block().bfloat16()
    if kind == "cnn":
        return make_cnn().bfloat16()
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.SiLU(), torch.nn.Linear(128, 10)
    ).bfloat16()


def batch(kind, gen):
    """(inputs, int64 targets) for model `kind`."""
    if kind == "block":
        return torch.randn(4, 32, 64, generator=gen).bfloat16(), torch.randint(
            0, 16, (4, 32), generator=gen
        )
    if kind == "cnn":
        return torch.randn(16, 3, 16, 16, generator=gen).bfloat16(), torch.randint(
            0, 10, (16,), generator=gen
        )
    return torch.randn(32, 64, generator=gen).bfloat16(), torch.randint(
        0, 10, (32,), generator=gen
    )


def train(device, steps, foreach, clip, kind="mlp"):
    """(losses, final weights on CPU) of `steps` AdamW steps on `device`."""
    model = make_model(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=foreach)
    x, y = (t.to(device) for t in batch(kind, torch.Generator().manual_seed(1)))
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), y.flatten())
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses, [p.detach().cpu() for p in model.parameters()]


def main():
    """Trains on CPU and on tt and prints one comparison record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--foreach", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--model", choices=["mlp", "block", "cnn"], default="mlp")
    args = parser.parse_args()
    run = (args.steps, args.foreach, args.clip, args.model)

    cpu_losses, cpu_weights = train("cpu", *run)
    kohakutt.open(0)
    try:
        before = kohakutt.counters()
        tt_losses, tt_weights = train("tt", *run)
        after = kohakutt.counters()
    finally:
        kohakutt.close()

    curve = [abs(a - b) / max(abs(b), 1e-6) for a, b in zip(tt_losses, cpu_losses)]
    weights = [diff_stats(t, c) for t, c in zip(tt_weights, cpu_weights)]
    print(
        json.dumps(
            {
                "model": args.model,
                "steps": args.steps,
                "foreach": args.foreach,
                "clip": args.clip,
                "cpu_loss": [
                    round(v, 4) for v in cpu_losses[:: max(1, args.steps // 5)]
                ],
                "tt_loss": [round(v, 4) for v in tt_losses[:: max(1, args.steps // 5)]],
                "max_rel_loss_dev": round(max(curve), 4),
                "final_rel_loss_dev": round(curve[-1], 4),
                "weights": weights,
                "transfers": {
                    k: after[k] - before[k] for k in ("uploads", "downloads")
                },
                "fallbacks": kohakutt.fallback_counts(),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
