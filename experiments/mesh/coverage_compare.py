"""Compares two Kohaku-TT coverage records (kohakutt.tools.coverage --out files) op by
op: outcome counts, and per-sample error statistics of the second record next to the
first's.

    python experiments/mesh/coverage_compare.py one_card.json mesh.json
    python experiments/mesh/coverage_compare.py one_card.json mesh.json --all \\
        # every op
"""

import argparse
import json
from pathlib import Path


def _load(path):
    return {r["op"]: r for r in json.loads(Path(path).read_text())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--all", action="store_true", help="also print unchanged ops")
    parser.add_argument("--limit", type=int, default=60, help="sample rows printed")
    args = parser.parse_args()
    base, other = _load(args.base), _load(args.other)
    keys = ("ok", "unsupported", "error")
    totals = {k: [0, 0] for k in keys}
    changed = 0
    for op in sorted(set(base) & set(other)):
        a, b = base[op], other[op]
        for k in keys:
            totals[k][0] += a[k]
            totals[k][1] += b[k]
        same = all(a[k] == b[k] for k in keys)
        if not same:
            changed += 1
        if args.all or not same:
            print(
                f"{op:32s} "
                + "  ".join(f"{k} {a[k]}->{b[k]}" for k in keys)
                + f"  max abs {a['max_abs']} -> {b['max_abs']}"
                + f"  min cosine {a['min_cosine']} -> {b['min_cosine']}"
            )
    print(
        f"ops compared {len(set(base) & set(other))}, outcome counts changed {changed}"
    )
    for k in keys:
        print(f"total {k}: {totals[k][0]} -> {totals[k][1]}")

    # Per sample (same op, same sample index): device error statistics against the
    # ruler, first record vs second.
    same_stats, diff_stats, paired = 0, [], 0
    for op in sorted(set(base) & set(other)):
        for i, (a, b) in enumerate(zip(base[op]["samples"], other[op]["samples"])):
            if "stats" not in a or "stats" not in b:
                continue
            paired += 1
            if a["stats"] == b["stats"]:
                same_stats += 1
            else:
                diff_stats.append((op, i, a, b))
    print(f"samples with stats in both: {paired}, identical stats: {same_stats}")
    for op, i, a, b in diff_stats[: args.limit]:
        sa, sb = a["stats"], b["stats"]
        print(
            f"  {op} #{i} {a.get('dtype')} {a.get('shapes')}: abs max "
            f"{sa['abs_max']:.3e} -> {sb['abs_max']:.3e}, "
            f"rel max {sa['rel_max']:.3e} -> "
            f"{sb['rel_max']:.3e}, cos {sa['cosine']:.6f} -> {sb['cosine']:.6f}, "
            f"nonfinite {sa['nonfinite']} -> {sb['nonfinite']}; cpu baseline abs max "
            + (f"{a['cpu_stats']['abs_max']:.3e}" if a.get("cpu_stats") else "(none)")
        )
    if len(diff_stats) > args.limit:
        print(f"  ... {len(diff_stats) - args.limit} more")
    # Second record's abs max over the first's, for samples whose stats differ.
    ratios = []
    families = {}
    for op, _, a, b in diff_stats:
        first, second = a["stats"]["abs_max"], b["stats"]["abs_max"]
        r = second / first if first > 0 else (1.0 if second == 0 else float("inf"))
        ratios.append(r)
        families.setdefault(op.split("#")[0].split(".")[0], []).append(r)
    if ratios:
        ratios.sort()
        bounds = [0.5, 0.9, 1.1, 1.5, 2.0, 4.0]
        counts = [
            sum(1 for r in ratios if lo <= r < hi)
            for lo, hi in zip([0.0, *bounds], [*bounds, float("inf")], strict=True)
        ]
        print(
            "abs max ratio (second / first) of differing samples: median "
            f"{ratios[len(ratios) // 2]:.3f}, max {ratios[-1]:.3f}; buckets "
            + ", ".join(
                f"[{lo:g}, {hi:g}): {n}"
                for lo, hi, n in zip(
                    [0.0, *bounds], [*bounds, float("inf")], counts, strict=True
                )
            )
        )
        for fam, rs in sorted(families.items(), key=lambda kv: -max(kv[1])):
            print(f"  {fam:24s} n {len(rs):3d}  max ratio {max(rs):.3f}")
    print(f"only in {args.base}: {len(set(base) - set(other))}")
    print(f"only in {args.other}: {len(set(other) - set(base))}")


if __name__ == "__main__":
    main()
