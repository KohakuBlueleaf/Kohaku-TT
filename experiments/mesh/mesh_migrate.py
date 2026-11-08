"""Placement migration (KTT_MESH_MIGRATE=1, set here) on a 1x2 mesh with automatic
placement (no other user calls):
- MLP blocks (1024 -> 4096 -> 1024, bf16), forward steps: the down projection's weight
  placement and the conversions per step;
- a tied embedding / lm_head model trained 100 SGD steps: placement changes of the
  shared weight;
- an MLP trained 20 AdamW steps: placement changes per parameter, and the loss against
  the same run with KTT_MESH_AUTO=0 (replicated).

    python experiments/mesh/mesh_migrate.py
"""

import itertools
import os

os.environ.setdefault("KTT_MESH_MIGRATE", "1")

import torch

import kohakutt


class MLP(torch.nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.up = torch.nn.Linear(dim, hidden)
        self.down = torch.nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


class Tied(torch.nn.Module):
    def __init__(self, vocab=8192, dim=512):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.mlp = MLP(dim, 2048)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        return self.head(self.mlp(self.embed(ids)))


def placements(model):
    return {n: kohakutt.placement(p) for n, p in model.named_parameters()}


def mlp_forward():
    torch.manual_seed(0)
    model = torch.nn.Sequential(MLP(), MLP()).to(torch.bfloat16).to("tt")
    x = torch.randn(256, 1024, dtype=torch.bfloat16).to("tt")
    for step in range(6):
        torch.tt.reset_mesh_stats()
        with torch.no_grad():
            model(x)
        stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
        print(
            f"mlp step {step}: down.weight "
            f"{kohakutt.placement(model[0].down.weight)} / "
            f"{kohakutt.placement(model[1].down.weight)}, up.weight "
            f"{kohakutt.placement(model[0].up.weight)}; {stats}"
        )


def tied_training():
    torch.manual_seed(0)
    model = Tied().to(torch.bfloat16).to("tt")
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 8192, (4, 64)).to("tt")
    history, changes = [], 0
    for _ in range(100):
        loss = model(ids).float().logsumexp(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        p = kohakutt.placement(model.embed.weight)
        changes += bool(history) and p != history[-1]
        history.append(p)
    print(
        f"tied: shared weight placements {sorted(set(history))}, "
        f"changes over 100 steps "
        f"{changes}, final {history[-1]}, "
        f"migrations {torch.tt.mesh_stats()['migrations']}"
    )


def adamw_training():
    losses = {}
    for mode in ("replicated", "auto"):
        os.environ["KTT_MESH_AUTO"] = "0" if mode == "replicated" else "1"
        torch.manual_seed(0)
        model = MLP().to(torch.bfloat16).to("tt")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(256, 1024, dtype=torch.bfloat16).to("tt")
        seen = {n: [p] for n, p in placements(model).items()}
        torch.tt.reset_mesh_stats()
        for _ in range(20):
            loss = model(x).float().pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            for n, p in placements(model).items():
                seen[n].append(p)
        losses[mode] = loss.item()
        if mode == "auto":
            for n, ps in seen.items():
                changes = sum(a != b for a, b in itertools.pairwise(ps))
                print(
                    f"adamw {n}: start {ps[0]}, final {ps[-1]}, placement changes "
                    f"{changes}"
                )
            print(f"adamw fallbacks per op: {torch.tt.mesh_fallback_ops()}")
    os.environ.pop("KTT_MESH_AUTO", None)
    print(
        f"adamw loss after 20 steps: replicated {losses['replicated']:.6f}, "
        f"auto {losses['auto']:.6f}"
    )


def main():
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        mlp_forward()
        tied_training()
        adamw_training()
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
