"""Lanes: one sender row multicasting reserved pages to the receiver rows of its column,
with the handshake (ready count, data, flush, valid flag) written as statements."""

from kohakutt.dsl.metal.binding import as_expr
from kohakutt.dsl.metal.body import Barrier, Mcast, McastSet, SemOp, Slot

# Values of the valid flag (tt-metal's VALID / INVALID).
VALID, INVALID = 1, 0


class Lane:
    """A lane of program `p.lane(senders, receivers)`: the receivers' rectangle
    `rect` (per sender column), the sender's place `head` (per receiver) and its
    semaphores `ready` (receivers that freed their slot, on the sender) and `flag`
    (the data is there, on each receiver)."""

    def __init__(self, name, senders, receivers, rect, head, ready, flag):
        self.name, self.senders, self.receivers = name, senders, receivers
        self.rect, self.head, self.ready, self.flag = rect, head, ready, flag

    def open(self, k):
        """Sender kernel `k`: its own valid flag, the value the lane multicasts."""
        k._lane(SemOp("set", self.flag, as_expr(VALID), lane=self))

    def send(self, k, slot):
        """Sender kernel `k`: waits until every receiver reserved its slot, then
        multicasts the whole reserve `slot` and, after the flush, the flag."""
        if not isinstance(slot, Slot):
            raise TypeError("kohakutt.dsl.metal: a lane sends a whole reserve")
        size = as_expr(slot.pages) * slot.cb.page_bytes
        k._lane(SemOp("wait_eq", self.ready, self.rect.count, lane=self))
        k._lane(SemOp("set", self.ready, as_expr(0), lane=self))
        k._lane(Mcast(slot[0], self.rect, size, self))
        k._add(Barrier("flushed"))
        k._lane(McastSet(self.flag, self.rect, self))

    def receive(self, k):
        """Receiver kernel `k`, holding the reserve the data lands in: clears its
        flag, tells the sender, and waits for the data."""
        k._lane(SemOp("set", self.flag, as_expr(INVALID), lane=self))
        k._lane(SemOp("inc", self.ready, as_expr(1), self.head, lane=self))
        k._lane(SemOp("wait_eq", self.flag, as_expr(VALID), lane=self))
