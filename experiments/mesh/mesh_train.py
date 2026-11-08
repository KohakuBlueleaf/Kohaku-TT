"""One training step of a small MLP block on a 1x2 mesh with sharded parameters
(up projection column-parallel, down projection row-parallel), against the same step
with replicated parameters (single-card kernels) and a CPU float64 / float32 model.

Prints, per parameter, error statistics of its gradient and of its value after one SGD
step against float64, for the sharded run, the replicated run and CPU float32; and the
sharded step's conversions (torch.tt.mesh_stats).

    python experiments/mesh/mesh_train.py
"""

import copy

import torch

import kohakutt


class Block(torch.nn.Module):
    def __init__(self, d=256, h=1024):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.up = torch.nn.Linear(d, h)
        self.down = torch.nn.Linear(h, d)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


def _step(model, x, lr=0.1):
    out = model(x)
    loss = out.square().mean()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad
    return loss.detach(), grads


def main():
    torch.manual_seed(0)
    ref = Block()
    x = torch.randn(64, 256)
    cpu64 = copy.deepcopy(ref).double()
    cpu32 = copy.deepcopy(ref)
    loss64, g64 = _step(cpu64, x.double())
    loss32, g32 = _step(cpu32, x)

    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        rep = copy.deepcopy(ref).to("tt")
        shd = copy.deepcopy(ref).to("tt")
        with torch.no_grad():
            shd.up.weight.data = kohakutt.shard(shd.up.weight.data, 0)
            shd.up.bias.data = kohakutt.shard(shd.up.bias.data, 0)
            shd.down.weight.data = kohakutt.shard(shd.down.weight.data, 1)
        print({n: kohakutt.placement(p) for n, p in shd.named_parameters()})
        loss_r, g_r = _step(rep, x.to("tt"))
        torch.tt.reset_mesh_stats()
        loss_s, g_s = _step(shd, x.to("tt"))
        stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
        print(f"sharded step conversions: {stats}")
        print(f"sharded step fallbacks per op: {torch.tt.mesh_fallback_ops()}")
        print(
            f"loss: float64 {loss64.item():.8f} float32 {loss32.item():.8f} "
            f"replicated {loss_r.cpu().item():.8f} sharded {loss_s.cpu().item():.8f}"
        )
        params = {
            "rep": dict(rep.named_parameters()),
            "shd": dict(shd.named_parameters()),
            "cpu32": dict(cpu32.named_parameters()),
        }
        ref_params = dict(cpu64.named_parameters())
        for name in g64:
            print(f"{name}: placement {kohakutt.placement(params['shd'][name])}")
            for tag, grads in (("sharded", g_s), ("replicated", g_r), ("cpu32", g32)):
                s = kohakutt.diff_stats(grads[name].cpu(), g64[name])
                print(f"    grad  {tag:10s} {kohakutt.format_stats(s)}")
            for tag in ("shd", "rep", "cpu32"):
                value = params[tag][name].detach().cpu()
                s = kohakutt.diff_stats(value, ref_params[name].detach())
                print(f"    value {tag:10s} {kohakutt.format_stats(s)}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
