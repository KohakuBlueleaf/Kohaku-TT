"""Host zone sequence of the dispatch worker during the last forwards of a
tracy-csvexport -u dump: per forward, the count and total time of each host zone on the
thread that issues EnqueueProgram, and the ordered zone list of the last forward.

    python experiments/mesh/mesh_tracy_ops.py out/devprof/p2048_dp/zones.csv \\
        --forwards 2
"""

import argparse
import collections
import csv

SKIP = ("-FW", "-KERNEL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zones")
    parser.add_argument("--forwards", type=int, default=2)
    parser.add_argument("--split-us", type=float, default=3000.0)
    parser.add_argument("--list", type=int, default=80)
    args = parser.parse_args()
    rows = []
    with open(args.zones) as f:
        for row in csv.DictReader(f):
            if row["name"].endswith(SKIP):
                continue
            s = int(row["ns_since_start"])
            rows.append(
                (
                    s,
                    s + int(row["exec_time_ns"]),
                    row["name"],
                    row["thread"],
                    row["zone_text"],
                )
            )
    rows.sort()
    worker = collections.Counter(
        t for _, _, n, t, _ in rows if n == "EnqueueProgram"
    ).most_common(1)[0][0]
    mine = [r for r in rows if r[3] == worker]
    enq = [r for r in mine if r[2] == "EnqueueProgram"]
    bursts = [[enq[0]]]
    for r in enq[1:]:
        if r[0] - bursts[-1][-1][1] > args.split_us * 1e3:
            bursts.append([])
        bursts[-1].append(r)
    print(f"worker thread {worker}; {len(bursts)} enqueue bursts")
    for b in bursts[-args.forwards :]:
        lo, hi = b[0][0] - 200e3, b[-1][1]
        zone = [r for r in mine if lo <= r[0] <= hi]
        count, total = collections.Counter(), collections.Counter()
        for s, e, n, _, _ in zone:
            count[n] += 1
            total[n] += e - s
        print(f"-- forward: {len(b)} enqueues over {(hi - lo) / 1e6:.3f} ms")
        for n, ns in total.most_common(12):
            print(
                f"   {count[n]:4d} x {ns / count[n] / 1e3:7.1f} us = "
                f"{ns / 1e6:6.3f} ms  {n}"
            )
    last = bursts[-1]
    lo, hi = last[0][0] - 200e3, last[-1][1]
    print("-- last forward, worker zones in order (start us, dur us)")
    t0 = None
    for s, e, n, _, text in [r for r in mine if lo <= r[0] <= hi][: args.list]:
        t0 = s if t0 is None else t0
        print(f"   {(s - t0) / 1e3:8.1f} {(e - s) / 1e3:7.1f}  {n} {text[:60]}")


if __name__ == "__main__":
    main()
