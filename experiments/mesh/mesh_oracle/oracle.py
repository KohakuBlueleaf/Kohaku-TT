"""Mesh placement oracle: DTensor's placement choices for representative ops on 1x2,
2x2 and 2x4 meshes, on the CPU with no card (torch's LocalTensorMode over the `fake`
process group).

For each op, mesh and input placement (per mesh axis: R, S(d) or P(sum); axis 0 of a
1x2 mesh is R), it records:

- DTensor's output placement;
- the input redistributions DTensor ran (source -> target placement per input) and the
  collectives they issued, by kind (CommDebugMode);
- error statistics of the assembled output against a float64 dense computation, next
  to the dense float32 computation's (the same-dtype baseline);
- the known intentional divergence of Kohaku-TT's mesh rules from DTensor's for that
  row: one-chip bias add on Partial, statistics norms and softmax over a sharded dim,
  equal padded shards for uneven sizes.

    python experiments/mesh/mesh_oracle/oracle.py                     # every case
    python experiments/mesh/mesh_oracle/oracle.py --meshes 2x2 --ops mm
    python experiments/mesh/mesh_oracle/oracle.py --json oracle.json --md oracle.md

Pinned to the torch it was written against (PINNED_TORCH); another version prints a
warning, since DTensor's strategies change between releases.
"""

import argparse
import itertools
import json
import warnings
from collections import Counter

import torch
import torch.distributed as dist
import torch.distributed.tensor._dispatch as dtensor_dispatch
import torch.nn.functional as F
from torch.distributed._local_tensor import (
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
    rank_map,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.distributed.fake_pg import FakeStore

PINNED_TORCH = "2.15.0+cpu"

MESHES = {"1x2": (1, 2), "2x2": (2, 2), "2x4": (2, 4)}


def stats(result, reference):
    """Error statistics of `result` against `reference`, in float64: abs and rel
    (over nonzero reference positions) max / median / mean, cosine, non-finite
    count."""
    r = result.double().flatten()
    ref = reference.double().flatten()
    err = (r - ref).abs()
    nz = ref != 0
    rel = err[nz] / ref[nz].abs() if nz.any() else torch.zeros(1, dtype=torch.double)
    cos = F.cosine_similarity(r, ref, dim=0, eps=1e-300) if r.numel() else 1.0
    return {
        "abs_max": err.max().item() if err.numel() else 0.0,
        "abs_median": err.median().item() if err.numel() else 0.0,
        "abs_mean": err.mean().item() if err.numel() else 0.0,
        "rel_max": rel.max().item(),
        "rel_median": rel.median().item(),
        "rel_mean": rel.mean().item(),
        "cosine": float(cos),
        "nonfinite": int((~torch.isfinite(r)).sum()),
    }


class Op:
    """An op: fn(*tensors), the dense inputs make(generator), the dims each input may
    shard (None: tensor not shardable, e.g. a scalar), whether each input may be
    Partial, and its divergence rule: divergence(placements_in, mesh_shape) -> note
    or None."""

    def __init__(self, name, family, fn, make, shard_dims, partial, divergence=None):
        self.name, self.family, self.fn, self.make = name, family, fn, make
        self.shard_dims, self.partial = shard_dims, partial
        self.divergence = divergence or (lambda pin, mesh_shape: None)


def _f(*shape):
    return lambda g: torch.randn(*shape, generator=g)


def _idx(n, vocab):
    return lambda g: torch.randint(0, vocab, (n,), generator=g)


def _is_p(pl):
    return isinstance(pl, Partial)


def _bias_divergence(pin, mesh_shape):
    """add(P, R): Kohaku-TT keeps P and adds R on coordinate 0 of each partial axis
    (no collective, exact); DTensor's rule differs."""
    a, b = pin
    for axis in range(len(a)):
        if _is_p(a[axis]) != _is_p(b[axis]) and (
            isinstance(a[axis], Replicate) or isinstance(b[axis], Replicate)
        ):
            return "one-chip bias add: Kohaku-TT keeps P, adds R on coordinate 0"
    return None


def _scalar_divergence(pin, mesh_shape):
    if any(_is_p(p) for p in pin[0]):
        return "scalar add on P: Kohaku-TT keeps P, adds on coordinate 0"
    return None


def _stats_divergence(dim):
    """Norms / softmax over `dim`: Kohaku-TT keeps S(dim) with partial statistics and
    a small all-reduce; DTensor gathers."""

    def rule(pin, mesh_shape):
        if any(isinstance(p, Shard) and p.dim == dim for p in pin[0]):
            return "statistics over sharded dim: Kohaku-TT keeps S, all-reduces stats"
        return None

    return rule


def _uneven_divergence(shape):
    """Uneven shards: Kohaku-TT pads every shard to one size and records `valid`
    (its extent before padding); DTensor uses torch.chunk sizes."""

    def rule(pin, mesh_shape):
        for pls in pin:
            for axis, p in enumerate(pls):
                if isinstance(p, Shard) and shape[p.dim] % mesh_shape[axis]:
                    return "uneven shard: Kohaku-TT pads equal shards (valid extent)"
        return None

    return rule


def ops():
    """The representative op list."""
    ln16 = _stats_divergence(1)
    return [
        Op("add", "elementwise", torch.add, [_f(8, 16), _f(8, 16)], [[0, 1]] * 2, True),
        Op(
            "add_bias",
            "elementwise",
            torch.add,
            [_f(8, 16), _f(16)],
            [[0, 1], [0]],
            True,
            _bias_divergence,
        ),
        Op(
            "add_scalar",
            "elementwise",
            lambda x: x + 1.0,
            [_f(8, 16)],
            [[0, 1]],
            True,
            _scalar_divergence,
        ),
        Op("mul", "elementwise", torch.mul, [_f(8, 16), _f(8, 16)], [[0, 1]] * 2, True),
        Op("gelu", "elementwise", F.gelu, [_f(8, 16)], [[0, 1]], True),
        Op("mm", "matmul", torch.mm, [_f(8, 16), _f(16, 8)], [[0, 1]] * 2, True),
        Op(
            "linear",
            "matmul",
            F.linear,
            [_f(8, 16), _f(8, 16), _f(8)],
            [[0, 1], [0, 1], [0]],
            True,
        ),
        Op(
            "bmm",
            "matmul",
            torch.bmm,
            [_f(4, 8, 16), _f(4, 16, 8)],
            [[0, 1, 2]] * 2,
            True,
        ),
        Op("sum_dim0", "reduction", lambda x: x.sum(0), [_f(8, 16)], [[0, 1]], True),
        Op("sum_dim1", "reduction", lambda x: x.sum(1), [_f(8, 16)], [[0, 1]], True),
        Op("sum_all", "reduction", lambda x: x.sum(), [_f(8, 16)], [[0, 1]], True),
        Op("mean_dim1", "reduction", lambda x: x.mean(1), [_f(8, 16)], [[0, 1]], True),
        Op("amax_dim1", "reduction", lambda x: x.amax(1), [_f(8, 16)], [[0, 1]], True),
        Op(
            "embedding",
            "embedding",
            F.embedding,
            [_idx(8, 16), _f(16, 8)],
            [[0], [0, 1]],
            [False, True],
        ),
        Op(
            "layer_norm",
            "norm",
            lambda x: F.layer_norm(x, (16,)),
            [_f(8, 16)],
            [[0, 1]],
            True,
            ln16,
        ),
        Op(
            "rms_norm",
            "norm",
            lambda x: F.rms_norm(x, (16,)),
            [_f(8, 16)],
            [[0, 1]],
            True,
            ln16,
        ),
        Op(
            "softmax",
            "softmax",
            lambda x: torch.softmax(x, -1),
            [_f(8, 16)],
            [[0, 1]],
            True,
            ln16,
        ),
        Op(
            "log_softmax",
            "softmax",
            lambda x: torch.log_softmax(x, -1),
            [_f(8, 16)],
            [[0, 1]],
            True,
            ln16,
        ),
        Op(
            "view_split0",
            "view",
            lambda x: x.view(4, 2, 16),
            [_f(8, 16)],
            [[0, 1]],
            True,
        ),
        Op(
            "view_split1",
            "view",
            lambda x: x.view(8, 4, 4),
            [_f(8, 16)],
            [[0, 1]],
            True,
        ),
        Op(
            "reshape_flat",
            "view",
            lambda x: x.reshape(128),
            [_f(8, 16)],
            [[0, 1]],
            True,
        ),
        Op("transpose", "view", lambda x: x.t(), [_f(8, 16)], [[0, 1]], True),
        Op(
            "add_uneven",
            "uneven",
            torch.add,
            [_f(5, 10), _f(5, 10)],
            [[0, 1]] * 2,
            False,
            _uneven_divergence((5, 10)),
        ),
        Op(
            "sum_uneven",
            "uneven",
            lambda x: x.sum(0),
            [_f(5, 10)],
            [[0, 1]],
            False,
            _uneven_divergence((5, 10)),
        ),
        Op(
            "reshape_flat_uneven",
            "uneven",
            lambda x: x.reshape(50),
            [_f(5, 10)],
            [[0, 1]],
            False,
            _uneven_divergence((5, 10)),
        ),
    ]


def _name(p):
    if isinstance(p, Replicate):
        return "R"
    if isinstance(p, Shard):
        return f"S({p.dim})"
    if isinstance(p, Partial):
        return f"P({p.reduce_op})"
    return str(p)


def _names(pls):
    return "[" + ",".join(_name(p) for p in pls) + "]"


def _axis_options(dims, partial):
    out = [Replicate()] + [Shard(d) for d in dims]
    if partial:
        out.append(Partial())
    return out


def placements_for(op, mesh_shape):
    """Every input placement combination of `op` on a mesh of `mesh_shape`."""
    per_input = []
    partial = op.partial
    if not isinstance(partial, list):
        partial = [partial] * len(op.make)
    for dims, can_p in zip(op.shard_dims, partial, strict=True):
        axes = [
            [Replicate()] if size == 1 else _axis_options(dims, can_p)
            for size in mesh_shape
        ]
        per_input.append([tuple(c) for c in itertools.product(*axes)])
    return itertools.product(*per_input)


def _to_dtensor(t, mesh, pls):
    """`t` distributed as `pls`; on a Partial axis coordinate 0 holds the value and
    the other coordinates zeros (the sum over the axis is `t`)."""
    partial = [axis for axis, p in enumerate(pls) if _is_p(p)]
    base = [Replicate() if _is_p(p) else p for p in pls]
    d = distribute_tensor(t, mesh, base)
    if not partial:
        return d
    local = d.to_local()
    shape = mesh.shape

    def keep(rank):
        coords = (rank // shape[1], rank % shape[1])
        return all(coords[axis] == 0 for axis in partial)

    per_rank = getattr(local, "_local_tensors", None)

    def part(rank):
        value = per_rank[rank] if per_rank is not None else local
        return value * (1 if keep(rank) else 0)

    parts = rank_map(part)
    return DTensor.from_local(
        parts, mesh, list(pls), run_check=False, shape=t.shape, stride=t.stride()
    )


def _rank0(t):
    """Rank 0's tensor of a LocalTensor (every rank's after full_tensor())."""
    local = getattr(t, "_local_tensors", None)
    return local[0] if local is not None else t


def run_case(op, mesh, mesh_shape, pin, gen_seed=0):
    with maybe_disable_local_tensor_mode():
        g = torch.Generator().manual_seed(gen_seed)
        dense = [m(g) for m in op.make]
        ref64 = op.fn(*[t.double() if t.is_floating_point() else t for t in dense])
        ref32 = op.fn(*dense)
    inputs = [_to_dtensor(t, mesh, pls) for t, pls in zip(dense, pin, strict=True)]

    moves = []
    original = dtensor_dispatch.redistribute_local_tensor

    def recording(local, current, target, **kwargs):
        moves.append((_names(current.placements), _names(target.placements)))
        return original(local, current, target, **kwargs)

    rec = {"inputs": [_names(p) for p in pin]}
    note = op.divergence(pin, mesh_shape)
    if note:
        rec["divergence"] = note
    dtensor_dispatch.redistribute_local_tensor = recording
    comm = CommDebugMode()
    try:
        with comm:
            out = op.fn(*inputs)
    except Exception as e:  # noqa: BLE001 - DTensor's refusal is a result
        rec.update(error=f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return rec
    finally:
        dtensor_dispatch.redistribute_local_tensor = original
    counts = Counter()
    for k, v in comm.get_comm_counts().items():
        counts[getattr(k, "__name__", str(k)).split(".")[-1]] += v
    try:
        full = _rank0(out.full_tensor())
    except Exception as e:  # noqa: BLE001 - DTensor failing to assemble is a result
        rec.update(
            output=_names(out.placements),
            redistributions=[f"{a}->{b}" for a, b in moves],
            collectives=dict(counts),
            collective_count=sum(counts.values()),
            error=f"full_tensor: {type(e).__name__}: {str(e).splitlines()[0][:120]}",
        )
        return rec
    with maybe_disable_local_tensor_mode():
        error, baseline = stats(full, ref64), stats(ref32, ref64)
    rec.update(
        output=_names(out.placements),
        redistributions=[f"{a}->{b}" for a, b in moves],
        collectives=dict(counts),
        collective_count=sum(counts.values()),
        stats=error,
        baseline=baseline,
    )
    return rec


def run_mesh(label, mesh_shape, op_names):
    n = mesh_shape[0] * mesh_shape[1]
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=n)
    rows = []
    try:
        with LocalTensorMode(n):
            mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=("rows", "cols"))
            for op in ops():
                if op_names and not any(op.name.startswith(o) for o in op_names):
                    continue
                for pin in placements_for(op, mesh_shape):
                    rec = run_case(op, mesh, mesh_shape, pin)
                    rec.update(mesh=label, op=op.name, family=op.family)
                    rows.append(rec)
    finally:
        dist.destroy_process_group()
    return rows


def markdown(rows):
    """A table of every row: mesh, op, inputs, output, redistributions, collectives,
    output error vs float64 (abs max) next to the float32 dense baseline's, and the
    known divergence."""
    lines = [
        (
            "| mesh | op | inputs | DTensor output | redistributions | collectives "
            "| abs max | baseline abs max | divergence |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            coll = ", ".join(
                f"{k} {v}" for k, v in sorted(r.get("collectives", {}).items())
            )
            lines.append(
                f"| {r['mesh']} | {r['op']} | {' '.join(r['inputs'])} "
                f"| {r.get('output', '')} error: {r['error'].replace('|', '/')} "
                f"| {'; '.join(r.get('redistributions', []))} | {coll} | | "
                f"| {r.get('divergence', '')} |"
            )
            continue
        coll = ", ".join(f"{k} {v}" for k, v in sorted(r["collectives"].items()))
        lines.append(
            f"| {r['mesh']} | {r['op']} | {' '.join(r['inputs'])} | {r['output']} "
            f"| {'; '.join(r['redistributions'])} | {coll} "
            f"| {r['stats']['abs_max']:.2e} | {r['baseline']['abs_max']:.2e} "
            f"| {r.get('divergence', '')} |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--meshes", nargs="*", default=list(MESHES))
    p.add_argument("--ops", nargs="*", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    args = p.parse_args()
    if torch.__version__ != PINNED_TORCH:
        warnings.warn(
            f"mesh oracle: written against torch {PINNED_TORCH}, running "
            f"{torch.__version__}",
            stacklevel=1,
        )
    rows = []
    for label in args.meshes:
        rows += run_mesh(label, MESHES[label], args.ops)
    summary = Counter()
    for r in rows:
        summary[(r["mesh"], "error" if "error" in r else "ok")] += 1
    print(f"torch {torch.__version__}: {len(rows)} rows", dict(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"torch": torch.__version__, "rows": rows}, f, indent=1)
    if args.md:
        with open(args.md, "w") as f:
            f.write(markdown(rows))
    if not args.json and not args.md:
        print(markdown(rows))


if __name__ == "__main__":
    main()
