"""One MOSS-Image DiT forward on a 1x2 mesh with automatic placement (default) against
replicated placement (KTT_MESH_AUTO=0, the single-card kernels).

Prints the velocity's error statistics against a CPU float64 model, next to the CPU
bf16 model's; warm forward time per run; the auto-placed run's conversions and gather
fallbacks per op; and the placements of the largest parameters. bf16 weights (the mesh
rules skip block-float storage).

    python experiments/mesh/mesh_moss.py --model /path/to/MOSS-Image-Base \
        --examples /path/to/Kohaku-TT/examples/06_moss_image \
        --inputs moss_real_inputs.pt
"""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import torch

import kohakutt


def _forward_ms(model, x, t, text, rope, repeat):
    with torch.no_grad():
        model(x, t, text, rope)
        torch.tt.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            out = model(x, t, text, rope)
        torch.tt.synchronize()
    return out, (time.perf_counter() - start) * 1e3 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MOSS-Image dir (HF layout)")
    parser.add_argument("--examples", required=True, help="examples/06_moss_image dir")
    parser.add_argument("--inputs", required=True, help="saved real inputs (.pt)")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, args.examples)
    load_dit = importlib.import_module("dit").load_dit

    root = Path(args.model) / "transformer"
    real = torch.load(args.inputs)
    x, t, text = real["audit_x"], real["sigma"] * 1000, real["positive"]
    side = x.shape[-1]

    results = {}
    with torch.no_grad():
        for name, dtype in (("cpu64", torch.float64), ("cpu16", torch.bfloat16)):
            cpu = load_dit(root, "cpu", dtype)
            rope = cpu.rope_for(side, side, text.shape[1], "cpu")
            results[name] = cpu(x.to(dtype), t, text.to(dtype), rope).double()
            del cpu
    ruler = results.pop("cpu64")

    dev = kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for mode in ("replicated", "auto"):
            os.environ["KTT_MESH_AUTO"] = "0" if mode == "replicated" else "1"
            model = load_dit(root, dev)
            if mode == "auto":
                big = sorted(model.named_parameters(), key=lambda kv: -kv[1].numel())
                for name, p in big[:4]:
                    print(f"param {name} {tuple(p.shape)}: {kohakutt.placement(p)}")
            rope = model.rope_for(side, side, text.shape[1], dev)
            torch.tt.reset_mesh_stats()
            out, ms = _forward_ms(model, x.to(dev), t, text.to(dev), rope, args.repeat)
            stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
            fallbacks = torch.tt.mesh_fallback_ops()
            results[mode] = out.cpu().double()
            print(
                f"{mode}: forward {ms:.0f} ms "
                f"({side * 16} px, warm, {args.repeat} runs)"
            )
            print(f"    conversions {stats}")
            print(f"    fallbacks per op {fallbacks}")
            del model, out
    finally:
        os.environ.pop("KTT_MESH_AUTO", None)
        kohakutt.close()
    for mode in ("replicated", "auto", "cpu16"):
        s = kohakutt.diff_stats(results[mode], ruler)
        print(f"velocity {mode:10s} {kohakutt.format_stats(s)}")


if __name__ == "__main__":
    main()
