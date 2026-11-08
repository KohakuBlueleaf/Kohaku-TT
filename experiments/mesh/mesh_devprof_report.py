"""Per-chip program timeline from a tt-metal device-profiler CSV
(profile_log_device.csv).

    python experiments/mesh/mesh_devprof_report.py \\
        out/devprof_2c_2048/.logs/profile_log_device.csv

A program is one run host ID on one chip; its span runs from the earliest firmware
zone start to the latest firmware zone end over all its cores. Forwards are split at
device idle gaps over --split-us. Per forward and chip it prints the span, the busy
time (union of program spans), the idle time between programs, and the programs in
order with core count, duration and the gap before each.
"""

import argparse
import collections
import csv
import statistics


def load(path):
    """Program spans {(chip, run_id): [start, end, cores, kernel_us]} in cycles, and
    MHz."""
    with open(path) as f:
        first = f.readline()
        mhz = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        col = {h: i for i, h in enumerate(header)}
        spans = {}
        cores = collections.defaultdict(set)
        kernel = collections.defaultdict(dict)
        open_kernel = {}
        for row in reader:
            name = row[col["zone name"]].strip()
            kind = row[col["type"]].strip()
            if not (name.endswith(("-FW", "-KERNEL"))):
                continue
            chip = int(row[col["PCIe slot"]])
            run = int(row[col["run host ID"]])
            ts = int(row[col["time[cycles since reset]"]])
            core = (int(row[col["core_x"]]), int(row[col["core_y"]]))
            risc = row[col["RISC processor type"]].strip()
            key = (chip, run)
            if name.endswith("-FW"):
                s = spans.setdefault(key, [ts, ts])
                s[0] = min(s[0], ts)
                s[1] = max(s[1], ts)
                cores[key].add(core)
            elif kind == "ZONE_START":
                open_kernel[(key, core, risc)] = ts
            elif kind == "ZONE_END" and (key, core, risc) in open_kernel:
                dur = ts - open_kernel.pop((key, core, risc))
                k = kernel[key]
                k[risc] = max(k.get(risc, 0), dur)
    out = {
        key: [s[0], s[1], len(cores[key]), {r: v / mhz for r, v in kernel[key].items()}]
        for key, s in spans.items()
    }
    return out, mhz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--split-us", type=float, default=5000.0)
    parser.add_argument(
        "--detail", type=int, default=1, help="forwards to list per program"
    )
    args = parser.parse_args()
    spans, mhz = load(args.csv)
    chips = sorted({c for c, _ in spans})
    print(f"chip clock {mhz:.0f} MHz, programs {len(spans)}")
    for chip in chips:
        progs = sorted(
            (
                (s[0], s[1], run, s[2], s[3])
                for (c, run), s in spans.items()
                if c == chip
            )
        )
        forwards = [[progs[0]]]
        for p in progs[1:]:
            if (p[0] - forwards[-1][-1][1]) / mhz > args.split_us:
                forwards.append([])
            forwards[-1].append(p)
        print(f"== chip {chip}: {len(forwards)} bursts")
        summary = []
        for i, fw in enumerate(forwards):
            span = (fw[-1][1] - fw[0][0]) / mhz
            busy = 0.0
            cur_s, cur_e = fw[0][0], fw[0][1]
            for s, e, *_ in fw[1:]:
                if s > cur_e:
                    busy += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            busy = (busy + cur_e - cur_s) / mhz
            summary.append((span, busy, len(fw)))
            print(
                f"  burst {i}: {len(fw)} programs, span {span / 1e3:.3f} ms, "
                f"busy {busy / 1e3:.3f} ms, "
                f"idle {(span - busy) / 1e3:.3f} ms"
            )
        if len(summary) > 2:
            mid = summary[1:]
            print(
                f"  median over bursts 1..: span "
                f"{statistics.median(s for s, _, _ in mid) / 1e3:.3f} ms,"
                f" busy {statistics.median(b for _, b, _ in mid) / 1e3:.3f} ms"
            )
        for i, fw in enumerate(forwards[-args.detail :] if args.detail else []):
            print(f"  -- programs of burst {len(forwards) - args.detail + i}")
            prev_end = None
            for s, e, run, ncores, kern in fw:
                gap = 0.0 if prev_end is None else (s - prev_end) / mhz
                prev_end = e if prev_end is None else max(prev_end, e)
                kern_s = " ".join(f"{r[:2]}{v:.0f}" for r, v in sorted(kern.items()))
                print(
                    f"    run {run:6d} cores {ncores:3d} dur {(e - s) / mhz:8.1f} us"
                    f" gap {gap:8.1f} us  kernel max us [{kern_s}]"
                )


if __name__ == "__main__":
    main()
