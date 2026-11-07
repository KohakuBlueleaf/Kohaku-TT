"""Linear -> SiLU -> Linear on tt:0 (or through Python TT-NN with --ttnn, in its own
process): fp64 error and µs per forward with the per-op overhead breakdown.
"""

import argparse
import json
import time

import torch

import kohakutt
from kohakutt.core.stats import diff_stats

REPEAT = 50


def make_model(d, h, gen):
    """bf16 Linear(d, h) -> SiLU -> Linear(h, d) with seeded weights."""
    model = torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.SiLU(), torch.nn.Linear(h, d)
    )
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) / p.shape[-1] ** 0.5)
    return model.bfloat16()


def reference(model, x):
    """fp64 forward of the bf16 model on the bf16 input."""
    with torch.no_grad():
        return model.double()(x.double())


def per_forward_us(step, sync):
    """(host µs per forward without waiting, wall µs per forward incl. device)."""
    step()
    sync()
    start = time.perf_counter()
    for _ in range(REPEAT):
        step()
    host = (time.perf_counter() - start) * 1e6 / REPEAT
    sync()
    wall = (time.perf_counter() - start) * 1e6 / REPEAT
    return host, wall


def run_native(native, tokens, d, h):
    """One record: correctness, residency counters and the overhead breakdown."""
    gen = torch.Generator().manual_seed(tokens)
    model = make_model(d, h, gen)
    x = torch.randn(tokens, d, generator=gen).bfloat16()
    ref = reference(make_model(d, h, torch.Generator().manual_seed(tokens)), x)
    with torch.no_grad():
        cpu_stats = diff_stats(model(x), ref)

    model_tt = model.to("tt")
    xt = x.to("tt")
    before = native.counters()
    with torch.no_grad():
        y = model_tt(xt)
    forward = {k: v - before[k] for k, v in native.counters().items()}
    stats = diff_stats(y.cpu(), ref)

    def step():
        with torch.no_grad():
            model_tt(xt)

    c0 = native.counters()
    host, wall = per_forward_us(step, native.synchronize)
    c1 = native.counters()
    ops_per_forward = forward["ops"]
    ttnn_us = (c1["op_ns"] - c0["op_ns"]) / 1e3 / (REPEAT + 1)
    return {
        "mode": "native",
        "tokens": tokens,
        "d": d,
        "h": h,
        "device": str(y.device),
        "stats": stats,
        "cpu_bf16_stats": cpu_stats,
        "forward_counters": forward,
        "host_us": round(host, 1),
        "wall_us": round(wall, 1),
        "ttnn_host_us": round(ttnn_us, 1),
        "torch_side_us": round(host - ttnn_us, 1),
        "per_op_host_us": round(host / ops_per_forward, 1),
    }


def run_ttnn(tokens, d, h):
    """The same MLP through Python TT-NN (HiFi4, fp32 accumulation) on device 0."""
    # Optional dependency: Python TT-NN, needed only by this mode.
    import ttnn

    gen = torch.Generator().manual_seed(tokens)
    model = make_model(d, h, gen)
    x = torch.randn(tokens, d, generator=gen).bfloat16()
    ref = reference(make_model(d, h, torch.Generator().manual_seed(tokens)), x)
    w1, b1 = model[0].weight, model[0].bias
    w2, b2 = model[2].weight, model[2].bias

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        cfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        upload = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t = [ttnn.from_torch(v.float(), **upload) for v in (x, w1, w2)]
        bias = [ttnn.from_torch(v.float().reshape(1, -1), **upload) for v in (b1, b2)]

        def step():
            a = ttnn.linear(
                t[0], t[1], bias=bias[0], transpose_b=True, compute_kernel_config=cfg
            )
            return ttnn.linear(
                ttnn.silu(a),
                t[2],
                bias=bias[1],
                transpose_b=True,
                compute_kernel_config=cfg,
            )

        out = ttnn.to_torch(ttnn.get_device_tensors(step())[0])
        stats = diff_stats(out, ref)
        host, wall = per_forward_us(step, lambda: ttnn.synchronize_device(device))
    finally:
        ttnn.close_mesh_device(device)
    return {
        "mode": "ttnn",
        "tokens": tokens,
        "d": d,
        "h": h,
        "stats": stats,
        "host_us": round(host, 1),
        "wall_us": round(wall, 1),
        "per_op_host_us": round(host / 3, 1),
    }


def main():
    """Runs every token count in the chosen mode; prints one record each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 256])
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--h", type=int, default=1024)
    parser.add_argument("--ttnn", action="store_true")
    args = parser.parse_args()

    if args.ttnn:
        for tokens in args.tokens:
            print(json.dumps(run_ttnn(tokens, args.d, args.h)), flush=True)
        return

    kohakutt.open(0)
    try:
        for tokens in args.tokens:
            print(json.dumps(run_native(kohakutt, tokens, args.d, args.h)), flush=True)
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
