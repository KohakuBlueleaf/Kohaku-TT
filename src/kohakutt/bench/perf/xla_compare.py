"""The same workloads on Kohaku-TT (eager, torch.compile) or on tt-xla (torch-xla with
pjrt-plugin-tt), one stack per process: median ms per step after warm-up. Run as a file
(the xla venv has no kohakutt): python src/kohakutt/bench/perf/xla_compare.py --stack
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """A pre-norm GPT block: causal self-attention and a GELU MLP."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, -1).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    """Token embedding, `layers` blocks, final norm, tied output head."""

    def __init__(self, vocab=4096, d=512, heads=8, layers=4, seq=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.blocks = nn.Sequential(*(Block(d, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        x = self.embed(ids) + self.pos[: ids.shape[1]]
        return self.norm(self.blocks(x)) @ self.embed.weight.t()


def workloads():
    """{name: make}, make() -> (seeded CPU model, CPU inputs, train flag)."""

    def mlp():
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 10),
        )
        return model, (torch.randn(64, 512), torch.randint(0, 10, (64,))), True

    def gpt(train):
        def make():
            torch.manual_seed(0)
            ids = torch.randint(0, 4096, (8, 256))
            return TinyGPT().bfloat16(), (ids, torch.roll(ids, -1, 1)), train

        return make

    return {
        "mlp_train": mlp,
        "gpt_train (4x512, 8x256 tokens)": gpt(True),
        "gpt_infer (4x512, 8x256 tokens)": gpt(False),
    }


def step_fn(model, opt, inputs, train):
    """One step: forward (+ CE loss, backward, AdamW update when training)."""
    x, y = inputs

    def step():
        if not train:
            with torch.no_grad():
                return model(x)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
        )
        opt.zero_grad(set_to_none=False)
        loss.backward()
        opt.step()
        return loss.detach()

    return step


def run_stack(stack, steps, warmup):
    """{workload: {mode: record}} on `stack`."""
    if stack == "xla":
        # Optional dependency: torch-xla with the TT pjrt plugin (tt-xla venv).
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        xr.set_device_type("TT")
        device, modes = torch_xla.device(), ["lazy"]

        def sync():
            torch_xla.sync()
            xm.wait_device_ops()

    else:
        # Optional dependency: Kohaku-TT itself (this repo's venv).
        import kohakutt

        device, modes = kohakutt.device(0), ["eager", "compiled"]

        def sync():
            kohakutt.synchronize(device)

    results = {}
    for name, make in workloads().items():
        results[name] = {}
        for mode in modes:
            model, inputs, train = make()
            model = model.to(device)
            inputs = tuple(t.to(device) for t in inputs)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3) if train else None
            run = model
            if mode == "compiled":
                run = torch.compile(model, backend="ktt_tt_aot" if train else "ktt_tt")
            step = step_fn(run, opt, inputs, train)
            # Outputs stay alive through the sync (a lazy stack computes live results).
            for _ in range(warmup):
                out = step()
                if stack == "xla":
                    sync()
            sync()
            samples = []
            for _ in range(steps):
                start = time.perf_counter()
                out = step()
                sync()
                samples.append((time.perf_counter() - start) * 1e3)
            del out
            record = {"ms_per_step": round(statistics.median(samples), 2)}
            results[name][mode] = record
            print(json.dumps({name: {mode: record}}), flush=True)
    return results


def main():
    """Runs every workload on one stack; prints one JSON line per (workload, mode)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["kohakutt", "xla"], required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    run_stack(args.stack, args.steps, args.warmup)


if __name__ == "__main__":
    main()
