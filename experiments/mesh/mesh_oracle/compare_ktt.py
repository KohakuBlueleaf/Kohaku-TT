"""Kohaku-TT's mesh placements against the DTensor oracle, on a 1 x 2 mesh of two chips.

For each 1x2 row of an oracle JSON (oracle.py --json), the same dense inputs are placed
on the mesh as the row's input placements (torch.ops.ktt.distribute), the op runs, and
the row records Kohaku-TT's output placement and conversion counts (kohakutt.mesh_stats
deltas), DTensor's next to them, whether they agree, the row's known divergence, and
error statistics of the assembled output against the float64 dense op next to DTensor's.

    python experiments/mesh/mesh_oracle/compare_ktt.py oracle.json \\
        --json compare.json --md compare.md
"""

import argparse
import json
import re
from collections import Counter

import torch
from oracle import ops, stats

import kohakutt

# Kohaku-TT mesh_stats keys -> DTensor collective names.
KINDS = {
    "all_gathers": "all_gather_into_tensor",
    "all_reduces": "all_reduce",
    "reduce_scatters": "reduce_scatter_tensor",
}


def _ktt_placement(t):
    """Kohaku-TT's placement of `t` as the oracle's "[A,B]" (torch dims)."""
    raw = torch.ops.ktt.placement(t)
    lift = max(0, 2 - t.dim())
    items = []
    for m in re.finditer(r"R|S\((\d+)\)|P\((\w+)\)", raw):
        if m.group(1) is not None:
            items.append(f"S({int(m.group(1)) - lift})")
        else:
            items.append(m.group(0))
    if not items:  # replicated: no axes
        items = ["R"] * len(torch.ops.ktt.mesh_shape())
    return "[" + ",".join(items) + "]"


def _spec(pls):
    return pls.strip("[]")


def run_row(op, row, mesh):
    g = torch.Generator().manual_seed(0)
    dense = [m(g) for m in op.make]
    ref64 = op.fn(*[t.double() if t.is_floating_point() else t for t in dense])
    cpu_stats = stats(op.fn(*dense), ref64)
    placed = [
        torch.ops.ktt.distribute(t.to(mesh), _spec(p))
        for t, p in zip(dense, row["inputs"], strict=True)
    ]
    kohakutt.synchronize()
    torch.tt.reset_mesh_stats()
    rec = {"op": row["op"], "inputs": row["inputs"], "cpu_stats": cpu_stats}
    try:
        out = op.fn(*placed)
        kohakutt.synchronize()
        rec["ktt_output"] = _ktt_placement(out)
        counts = torch.tt.mesh_stats()
        rec["ktt_conversions"] = {
            k: v for k, v in counts.items() if v and k != "gather_bytes"
        }
        full = out.cpu()
        rec["ktt_stats"] = stats(full, ref64)
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        rec["ktt_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    rec["dtensor_output"] = row.get("output")
    rec["dtensor_collectives"] = row.get("collectives", {})
    rec["dtensor_error"] = row.get("error")
    rec["divergence"] = row.get("divergence")
    if "ktt_output" in rec and rec["dtensor_output"]:
        rec["same_output"] = rec["ktt_output"] == rec["dtensor_output"]
        ktt = Counter(
            {KINDS[k]: v for k, v in rec["ktt_conversions"].items() if k in KINDS}
        )
        dt = Counter(
            {k: v for k, v in rec["dtensor_collectives"].items() if k in KINDS.values()}
        )
        rec["same_collectives"] = ktt == dt
    return rec


def markdown(recs):
    lines = [
        (
            "| op | inputs | Kohaku-TT | conversions | DTensor | collectives | same | "
            "abs max (Kohaku-TT) | abs max (CPU same dtype) | divergence |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        conv = ", ".join(
            f"{k} {v}" for k, v in sorted(r.get("ktt_conversions", {}).items())
        )
        coll = ", ".join(
            f"{k} {v}" for k, v in sorted(r["dtensor_collectives"].items())
        )
        same = (
            ""
            if "same_output" not in r
            else f"{'out' if r['same_output'] else 'OUT'}/"
            f"{'comm' if r['same_collectives'] else 'COMM'}"
        )
        ktt = r.get("ktt_output") or f"error: {r.get('ktt_error', '')[:60]}"
        dt = r["dtensor_output"] or f"error: {(r['dtensor_error'] or '')[:60]}"
        absmax = f"{r['ktt_stats']['abs_max']:.2e}" if "ktt_stats" in r else ""
        cpu = f"{r['cpu_stats']['abs_max']:.2e}"
        lines.append(
            f"| {r['op']} | {' '.join(r['inputs'])} | {ktt} | {conv} | {dt} | {coll} "
            f"| {same} | {absmax} | {cpu} | {r.get('divergence') or ''} |".replace(
                "\n", " "
            )
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("oracle")
    p.add_argument("--ops", nargs="*", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    args = p.parse_args()
    with open(args.oracle) as f:
        rows = [r for r in json.load(f)["rows"] if r["mesh"] == "1x2"]
    by_name = {op.name: op for op in ops()}
    mesh = kohakutt.mesh([0, 1], shape=(1, 2))
    recs = []
    try:
        for row in rows:
            if args.ops and not any(row["op"].startswith(o) for o in args.ops):
                continue
            recs.append(run_row(by_name[row["op"]], row, mesh))
            r = recs[-1]
            print(
                f"{r['op']:20s} {' '.join(r['inputs']):24s} ktt "
                f"{r.get('ktt_output', 'error')} dtensor {r['dtensor_output']} "
                f"{r.get('same_output', '')} {r.get('same_collectives', '')}",
                flush=True,
            )
    finally:
        kohakutt.close()
    summary = Counter(
        (r.get("same_output"), r.get("same_collectives"), bool(r.get("divergence")))
        for r in recs
    )
    print("(same output, same collectives, known divergence): count", dict(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(recs, f, indent=1)
    if args.md:
        with open(args.md, "w") as f:
            f.write(markdown(recs))


if __name__ == "__main__":
    main()
