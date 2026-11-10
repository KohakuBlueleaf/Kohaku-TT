"""The compute grid a program is laid out on, and the core counts it may run on."""

from dataclasses import dataclass

# Core counts a program may be built for, in steps of at most 1.5x; "grid" is every
# core.
SIZE_CLASSES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, "grid")
# Bytes of L1 per core free for a program's buffers on an empty chip, less the
# native side's 32 KiB margin: a build-time `l1_budget`.
L1_BUDGET = 1362816 - 32 * 1024


@dataclass(frozen=True)
class Machine:
    """One chip's Tensix grid as kernels see it: logical columns and rows."""

    cols: int = 11
    rows: int = 10

    @property
    def cores(self):
        """The number of cores."""
        return self.cols * self.rows

    def classes(self, candidates=SIZE_CLASSES):
        """The core counts of `candidates` that fit the grid, ascending."""
        counts = {self.cores if c == "grid" else int(c) for c in candidates}
        return sorted(c for c in counts if 1 <= c <= self.cores)

    def pick(self, wanted, candidates=SIZE_CLASSES):
        """The largest core count of `candidates` not above `wanted` (at least 1)."""
        classes = self.classes(candidates)
        fitting = [c for c in classes if c <= max(1, wanted)]
        return fitting[-1] if fitting else classes[0]

    def core_list(self, count):
        """The first `count` cores, row-major, as (x, y)."""
        return [(i % self.cols, i // self.cols) for i in range(count)]


class CoreSet:
    """Cores of a program: every core it runs on, or those in rows first..last."""

    def __init__(self, first=None, last=None):
        self.first, self.last = first, last

    def rows(self, first, last=None):
        """The program's cores in grid rows `first`..`last` (one row by default)."""
        return CoreSet(first, first if last is None else last)

    def covers(self, other):
        """True when every core of `other` is in this set, whatever the core count."""
        if self.first is None:
            return True
        if other.first is None:
            return False
        return self.first <= other.first and other.last <= self.last

    def overlaps(self, other):
        """True when the two sets can share a core."""
        if self.first is None or other.first is None:
            return True
        return self.first <= other.last and other.first <= self.last

    def select(self, cores):
        """The cores of `cores` in this set."""
        if self.first is None:
            return list(cores)
        return [c for c in cores if self.first <= c[1] <= self.last]


def ranges(cores):
    """`cores` as [x0, y0, x1, y1] rectangles: runs within a row, equal runs merged
    down consecutive rows."""
    runs = []
    for x, y in sorted(set(cores), key=lambda c: (c[1], c[0])):
        if runs and runs[-1][3] == y and runs[-1][2] == x - 1:
            runs[-1][2] = x
        else:
            runs.append([x, y, x, y])
    out = []
    for run in runs:
        above = out[-1] if out else None
        if (
            above
            and above[0] == run[0]
            and above[2] == run[2]
            and above[3] == run[1] - 1
        ):
            above[3] = run[3]
        else:
            out.append(run)
    return out
