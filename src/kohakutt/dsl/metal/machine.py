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


def _inside(span, other):
    """True when interval `other` lies in `span` (None: unbounded)."""
    if span[0] is None:
        return True
    return other[0] is not None and span[0] <= other[0] and other[1] <= span[1]


def _meets(a, b):
    """True when intervals `a` and `b` (None: unbounded) can share a value."""
    return a[0] is None or b[0] is None or (a[0] <= b[1] and b[0] <= a[1])


class CoreSet:
    """Cores of a program: every core it runs on, or those in rows first..last and
    (`col_first`, `col_last` not None) columns col_first..col_last."""

    def __init__(self, first=None, last=None, col_first=None, col_last=None):
        self.first, self.last = first, last
        self.col_first, self.col_last = col_first, col_last

    @property
    def whole(self):
        """True for every core of the program."""
        return self.first is None and self.col_first is None

    def rows(self, first, last=None):
        """This set's cores in grid rows `first`..`last` (one row by default)."""
        last = first if last is None else last
        return CoreSet(first, last, self.col_first, self.col_last)

    def cols(self, first, last=None):
        """This set's cores in grid columns `first`..`last` (one by default)."""
        last = first if last is None else last
        return CoreSet(self.first, self.last, first, last)

    def covers(self, other):
        """True when every core of `other` is in this set, whatever the core count."""
        return _inside((self.first, self.last), (other.first, other.last)) and _inside(
            (self.col_first, self.col_last), (other.col_first, other.col_last)
        )

    def overlaps(self, other):
        """True when the two sets can share a core."""
        return _meets((self.first, self.last), (other.first, other.last)) and _meets(
            (self.col_first, self.col_last), (other.col_first, other.col_last)
        )

    def holds(self, core):
        """True when core (x, y) is in this set."""
        x, y = core
        return (self.first is None or self.first <= y <= self.last) and (
            self.col_first is None or self.col_first <= x <= self.col_last
        )

    def select(self, cores):
        """The cores of `cores` in this set."""
        return [c for c in cores if self.holds(c)]

    def cells(self, machine):
        """The grid cells (x, y) of `machine` in this set."""
        grid = [(x, y) for y in range(machine.rows) for x in range(machine.cols)]
        return set(self.select(grid))


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
