"""Host dispatch against device execution from a tracy-csvexport -u dump (zones.csv).

    python experiments/mesh/mesh_tracy_report.py out/devprof/t2048/zones.csv \\
        --forwards 4

Device busy segments are the union of all *-FW zones (every core, both chips); host
enqueues are EnqueueProgram zones. For the last --forwards bursts (split at device idle
over --split-us) it prints the forward span, device busy, device idle, the idle that
ends within --lag-us of a host enqueue (host-bound idle), and on the enqueue thread the
time inside EnqueueProgram and the time between enqueues.
"""

import argparse
import bisect
import csv
import itertools
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zones")
    parser.add_argument("--forwards", type=int, default=4)
    parser.add_argument("--split-us", type=float, default=3000.0)
    parser.add_argument("--lag-us", type=float, default=25.0)
    args = parser.parse_args()
    dev, enq = [], []
    with open(args.zones) as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name.endswith("-FW"):
                s = int(row["ns_since_start"])
                dev.append((s, s + int(row["exec_time_ns"])))
            elif name == "EnqueueProgram":
                s = int(row["ns_since_start"])
                enq.append((s, s + int(row["exec_time_ns"]), row["thread"]))
    dev.sort()
    enq.sort()
    segs = [list(dev[0])]
    for s, e in dev[1:]:
        if s <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    bursts = [[segs[0]]]
    for seg in segs[1:]:
        if seg[0] - bursts[-1][-1][1] > args.split_us * 1e3:
            bursts.append([])
        bursts[-1].append(seg)
    enq_ends = [e for _, e, _ in enq]
    rows = []
    for b in bursts[-args.forwards :]:
        span = b[-1][1] - b[0][0]
        busy = sum(e - s for s, e in b)
        idle = span - busy
        host_bound = 0
        for (_, e0), (s1, _) in itertools.pairwise(b):
            i = bisect.bisect_right(enq_ends, s1) - 1
            if i >= 0 and enq_ends[i] > e0 and s1 - enq_ends[i] < args.lag_us * 1e3:
                host_bound += s1 - e0
        lo = bisect.bisect_left(enq_ends, b[0][0] - 2e6)
        hi = bisect.bisect_right(enq_ends, b[-1][1])
        mine = enq[lo:hi]
        inside = sum(e - s for s, e, _ in mine)
        between = [s1 - e0 for (_, e0, _), (s1, _, _) in itertools.pairwise(mine)]
        host_span = mine[-1][1] - mine[0][0] if mine else 0
        rows.append(
            (span, busy, idle, host_bound, len(mine), inside, sum(between), host_span)
        )
        print(
            f"burst: span {span / 1e6:.3f} ms  device busy {busy / 1e6:.3f}  "
            f"idle {idle / 1e6:.3f}"
            f" (host-bound {host_bound / 1e6:.3f})  | host: {len(mine)} enqueues over"
            f" {host_span / 1e6:.3f} ms, in EnqueueProgram {inside / 1e6:.3f},"
            f" between enqueues {sum(between) / 1e6:.3f} "
            f"(median {statistics.median(between) / 1e3:.1f} us)"
            if between
            else f"burst: span {span / 1e6:.3f} ms busy {busy / 1e6:.3f}"
        )


if __name__ == "__main__":
    main()
