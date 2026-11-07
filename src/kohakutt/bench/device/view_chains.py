"""Experiment: random view-op chains on tt vs CPU, bit-exact (data movement only).

Each chain applies 1-4 random view ops to a bf16 tensor on both devices, then compares
`.cpu()`; outcomes: ok, unsupported (contract error) or wrong (a bug).
    python -m kohakutt.bench.device.view_chains [--chains 200] [--seed 0]
"""

import argparse
import json
import random
from collections import Counter

import torch

import kohakutt


def random_op(rng, t):
    """(description, fn) of a random view op valid for tensor `t`, or None."""
    shape = list(t.shape)
    rank = len(shape)
    choices = []
    if rank >= 2:
        a, b = rng.sample(range(rank), 2)
        choices.append((f"transpose({a},{b})", lambda x: x.transpose(a, b)))
        perm = list(range(rank))
        rng.shuffle(perm)
        choices.append((f"permute{perm}", lambda x: x.permute(*perm)))
    if rank >= 1:
        d = rng.randrange(rank)
        if shape[d] > 1:
            start = rng.randrange(shape[d] - 1)
            length = rng.randrange(1, shape[d] - start + 1)
            choices.append(
                (f"narrow({d},{start},{length})", lambda x: x.narrow(d, start, length))
            )
            step = rng.choice([2, 3])
            idx = [slice(None)] * rank
            idx[d] = slice(None, None, step)
            choices.append((f"step({d},{step})", lambda x: x[tuple(idx)]))
            i = rng.randrange(shape[d])
            choices.append((f"select({d},{i})", lambda x: x.select(d, i)))
        choices.append(("flatten", lambda x: x.flatten()))
        if t.is_contiguous() and t.numel() % 2 == 0:
            choices.append(("view(2,-1)", lambda x: x.view(2, -1)))
        choices.append(("reshape(-1,1)", lambda x: x.reshape(-1, 1)))
    u = rng.randrange(rank + 1)
    choices.append((f"unsqueeze({u})", lambda x: x.unsqueeze(u)))
    if 1 in shape:
        s = shape.index(1)
        choices.append((f"squeeze({s})", lambda x: x.squeeze(s)))
        expanded = list(shape)
        expanded[s] = 3
        choices.append((f"expand{expanded}", lambda x: x.expand(*expanded)))
    return rng.choice(choices) if choices else None


def run_chain(rng, base_cpu):
    """One record: the chain and its outcome."""
    x_cpu, x_tt, steps = base_cpu, base_cpu.to("tt"), []
    try:
        for _ in range(rng.randint(1, 4)):
            op = random_op(rng, x_cpu)
            if op is None:
                break
            name, fn = op
            steps.append(name)
            x_cpu, x_tt = fn(x_cpu), fn(x_tt)
        got = x_tt.cpu()
    except NotImplementedError as error:
        return {"chain": steps, "outcome": "unsupported", "message": str(error)[:160]}
    if torch.equal(got, x_cpu.contiguous()):
        return {"chain": steps, "outcome": "ok"}
    return {"chain": steps, "outcome": "wrong", "shape": list(x_cpu.shape)}


def run_write_chain(rng, base_cpu):
    """One record: a random view chain, then copy_ of new values into the view; the
    whole base must match the same write on CPU."""
    x_cpu, x_tt = base_cpu.clone(), base_cpu.to("tt")
    v_cpu, v_tt, steps = x_cpu, x_tt, []
    try:
        for _ in range(rng.randint(1, 3)):
            op = random_op(rng, v_cpu)
            if op is None:
                break
            name, fn = op
            if name.startswith(("expand", "reshape", "flatten")):
                continue
            steps.append(name)
            v_cpu, v_tt = fn(v_cpu), fn(v_tt)
        values = torch.randn(v_cpu.shape).bfloat16()
        v_cpu.copy_(values)
        v_tt.copy_(values.to("tt"))
        got = x_tt.cpu()
    except NotImplementedError as error:
        return {"chain": steps, "outcome": "unsupported", "message": str(error)[:160]}
    if torch.equal(got, x_cpu):
        return {"chain": steps, "outcome": "ok"}
    return {"chain": steps, "outcome": "wrong", "shape": list(v_cpu.shape)}


def main():
    """Runs the chains and prints outcome counts plus every non-ok chain."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="write through the views")
    args = parser.parse_args()
    chain = run_write_chain if args.write else run_chain
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    kohakutt.open(0)
    try:
        records = []
        for _ in range(args.chains):
            shape = [
                rng.choice([1, 3, 5, 32, 40, 64]) for _ in range(rng.randint(1, 4))
            ]
            base = torch.randn(*shape, generator=gen).bfloat16()
            records.append({"base": shape, **chain(rng, base)})
    finally:
        kohakutt.close()
    print(json.dumps(Counter(r["outcome"] for r in records)))
    for r in records:
        if r["outcome"] != "ok":
            print(json.dumps(r))


if __name__ == "__main__":
    main()
