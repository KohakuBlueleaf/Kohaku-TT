"""TT-NN CCL inside a trace on a 1x2 mesh: functions whose rules issue all-gather,
all-reduce and reduce-scatter are captured (kohakutt.capture) and replayed on new
inputs; each replay is compared with the eager result bit for bit and against float64
next to the CPU same-dtype result, with the eager call's conversions.

    python experiments/mesh/mesh_trace.py
"""

import torch

import kohakutt


def gather_step(x, w):
    return (x @ w).sum(1)  # x S(1), w S(0): K-sharded matmul -> P(sum), then a sum


def reduce_step(x, w):
    return (x @ w).softmax(-1)  # P(sum) into softmax: all-reduce / reduce-scatter


def gelu_step(x, w):
    return torch.nn.functional.gelu(x @ w)  # P(sum) into an elementwise op


def rows_step(x, w):
    return (x.t() @ w).exp()  # x S(0) transposed: K-sharded, w partitioned


F32, BF16 = torch.float32, torch.bfloat16
K_SHARDED = (((64, 96), "R,S(1)"), ((96, 128), "R,S(0)"))
CASES = [
    ("K-sharded matmul + sum", gather_step, F32, *K_SHARDED),
    ("K-sharded matmul + softmax", reduce_step, F32, *K_SHARDED),
    ("K-sharded matmul + softmax", reduce_step, BF16, *K_SHARDED),
    ("K-sharded matmul + gelu", gelu_step, BF16, *K_SHARDED),
    (
        "transpose matmul + exp",
        rows_step,
        BF16,
        ((128, 64), "R,S(0)"),
        ((128, 128), "R,R"),
    ),
    (
        "transpose matmul + exp",
        rows_step,
        F32,
        ((96, 64), "R,S(0)"),
        ((96, 128), "R,R"),
    ),
]


def _inputs(dtype, spec_x, spec_w, seed):
    g = torch.Generator().manual_seed(seed)
    (sx, px), (sw, pw) = spec_x, spec_w
    x = (torch.randn(sx, generator=g) / 4).to(dtype)
    w = (torch.randn(sw, generator=g) / 4).to(dtype)
    xd = torch.ops.ktt.distribute(x.to("tt"), px)
    wd = torch.ops.ktt.distribute(w.to("tt"), pw)
    return (x, w), (xd, wd)


def main():
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for label_case, fn, dtype, spec_x, spec_w in CASES:
            name = f"{label_case} {dtype}"
            _, example = _inputs(dtype, spec_x, spec_w, 0)
            torch.tt.reset_mesh_stats()
            fn(*example)
            torch.tt.synchronize()
            eager_stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
            try:
                cap = kohakutt.capture(fn, *example)
            except Exception as e:  # noqa: BLE001 - a refusal is a result
                print(f"{name}: capture failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            for seed in (1, 2):
                host, dev = _inputs(dtype, spec_x, spec_w, seed)
                replay = cap(*dev).cpu()
                eager = fn(*dev).cpu()
                ref = fn(*[t.double() for t in host])
                same = torch.equal(replay, eager)
                print(
                    f"{name} seed {seed}: replay == eager {same} (eager {eager_stats})"
                )
                for label, t in (("replay", replay), ("cpu", fn(*host))):
                    s = kohakutt.format_stats(kohakutt.diff_stats(t.double(), ref))
                    print(f"    {label:6s} {s}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
