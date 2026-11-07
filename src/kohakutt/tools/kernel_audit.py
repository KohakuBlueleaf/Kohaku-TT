"""Per-kernel audit: every aten call on tt tensors checked against its CPU simulation
(the op on the same inputs in fp64 = ruler, and in the call's dtypes = simulation;
kohakutt.diff_stats of both), or timed device-synchronized. Views and calls without
floating outputs are skipped."""

import collections
import time

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

import kohakutt
from kohakutt.core.stats import diff_stats, format_stats

aten = torch.ops.aten


def _sdpa_overrideable(
    q,
    k,
    v,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    scale=None,
):
    """CPU simulation of the tt-only fused attention overload (its first output)."""
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return (out,)


# tt-only overloads and their CPU simulation (same arguments).
_CPU_EQUIVALENT = {
    aten._scaled_dot_product_fused_attention_overrideable.default: _sdpa_overrideable,
}


def _to64(t):
    return t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t


def _meta(t):
    return t.to("meta") if isinstance(t, torch.Tensor) else t


def _upcast(t):
    is_low = isinstance(t, torch.Tensor) and t.dtype in (torch.bfloat16, torch.half)
    return t.float() if is_low else t


def simulate(func, args, kwargs):
    """func on CPU tensors as a low-precision kernel with fp32 internals computes it:
    bf16 / fp16 inputs upcast to fp32, floating outputs rounded to the dtypes func
    gives on the original inputs (from a meta run)."""
    try:
        shapes = tree_flatten(func(*tree_map(_meta, args), **tree_map(_meta, kwargs)))[
            0
        ]
    except Exception:  # noqa: BLE001 (no meta kernel: run as is)
        return func(*args, **kwargs)
    out = func(*tree_map(_upcast, args), **tree_map(_upcast, kwargs))
    flat, spec = tree_flatten(out)
    dtypes = [m.dtype if isinstance(m, torch.Tensor) else None for m in shapes]
    cast = [
        t.to(d) if isinstance(t, torch.Tensor) and d is not None and t.dtype != d else t
        for t, d in zip(flat, dtypes, strict=False)
    ]
    return spec.unflatten(cast)


class Simulation(TorchDispatchMode):
    """Runs CPU code with every op through simulate() (bf16 rounding per op, fp32 math);
    fast on CPUs without native bf16 arithmetic."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.is_view:
            return func(*args, **(kwargs or {}))
        return simulate(func, args, kwargs or {})


def _excess(c):
    """tt mean abs error over the simulation's for one checked call (sort key)."""
    return c["tt"]["abs_mean"] / max(c["sim"]["abs_mean"], 1e-30)


def _is_tt(x):
    return isinstance(x, torch.Tensor) and x.device.type not in ("cpu", "meta")


def _signature(args):
    """Shapes and dtypes of the tensor arguments, for grouping calls."""
    flat, _ = tree_flatten(args)
    return tuple(
        (tuple(t.shape), str(t.dtype).replace("torch.", ""))
        for t in flat
        if isinstance(t, torch.Tensor)
    )


def known_copies(tt_module, cpu_module):
    """{TensorImpl: CPU tensor} for KernelAudit(known=...): parameters and buffers of
    `tt_module` mapped to the same-named ones of an identical CPU module."""
    cpu = dict(cpu_module.named_parameters()) | dict(cpu_module.named_buffers())
    pairs = [*tt_module.named_parameters(), *tt_module.named_buffers()]
    return {t._cdata: cpu[name].detach() for name, t in pairs if name in cpu}


def error(result, ruler):
    """kohakutt.core.stats.diff_stats of result against ruler."""
    return diff_stats(result, ruler)


class KernelAudit(TorchDispatchMode):
    """Context manager: with KernelAudit("precision" | "timing") as audit: run(); then
    audit.table()."""

    def __init__(self, mode="precision", device=None, known=None):
        super().__init__()
        self.mode = mode
        self.device = device
        self.calls = []
        # CPU copies of persistent tt tensors (weights), keyed by TensorImpl.
        self.known = known or {}
        # Calls are checked only while active (see only_within()).
        self.active = True
        self._depth = 0

    def only_within(self, modules):
        """Checks only calls made inside `modules` (forward hooks toggle `active`);
        returns the hook handles."""
        self.active = False
        handles = []

        def enter(m, args):
            self._depth += 1
            self.active = True

        def leave(m, args, out):
            self._depth -= 1
            self.active = self._depth > 0

        for m in modules:
            handles.append(m.register_forward_pre_hook(enter))
            handles.append(m.register_forward_hook(leave))
        return handles

    def _host(self, t):
        if not _is_tt(t):
            return t
        cached = self.known.get(t._cdata)
        return cached if cached is not None else t.detach().cpu()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat, _ = tree_flatten((args, kwargs))
        if not self.active or func.is_view or not any(_is_tt(t) for t in flat):
            return func(*args, **kwargs)
        if self.mode == "timing":
            return self._timed(func, args, kwargs)
        return self._checked(func, args, kwargs)

    def _timed(self, func, args, kwargs):
        kohakutt.synchronize(self.device)
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        kohakutt.synchronize(self.device)
        self.calls.append(
            {
                "op": str(func),
                "sig": _signature(args),
                "ms": (time.perf_counter() - t0) * 1e3,
            }
        )
        return out

    def _checked(self, func, args, kwargs):
        host = tree_map(self._host, (args, kwargs))
        out = func(*args, **kwargs)
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        if not any(t.is_floating_point() for t in outs):
            return out
        cpu_func = _CPU_EQUIVALENT.get(func, func)
        record = {"op": str(func), "sig": _signature(args)}
        try:
            ruler = tree_flatten(
                cpu_func(*tree_map(_to64, host[0]), **tree_map(_to64, host[1]))
            )[0]
            sim = tree_flatten(simulate(cpu_func, host[0], host[1]))[0]
        except Exception as exc:  # noqa: BLE001 (any CPU failure: recorded, not fatal)
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            self.calls.append(record)
            return out
        ruler = [t for t in ruler if isinstance(t, torch.Tensor)]
        sim = [t for t in sim if isinstance(t, torch.Tensor)]
        tt_out, ruler_out, sim_out = outs[0], ruler[0], sim[0]
        if tt_out.is_floating_point() and tt_out.numel() > 0:
            got = tt_out.detach().cpu()
            record["tt"] = error(got, ruler_out)
            record["sim"] = error(sim_out, ruler_out)
            record["nonfinite"] = int((~torch.isfinite(got)).sum()) - int(
                (~torch.isfinite(ruler_out)).sum()
            )
        self.calls.append(record)
        return out

    def table(self, top=40):
        """Text report: timing per (op, signature) by total time, or precision per call
        group ranked by tt mean abs error over the simulation's."""
        groups = collections.defaultdict(list)
        for c in self.calls:
            groups[(c["op"], c["sig"])].append(c)
        lines = []
        if self.mode == "timing":
            total = sum(c["ms"] for c in self.calls)
            lines.append(f"total {total:.1f} ms over {len(self.calls)} calls")
            lines.append(
                f"{'op':48s} {'calls':>5s} {'ms':>8s} {'ms/call':>8s}  signature"
            )
            rows = sorted(groups.items(), key=lambda kv: -sum(c["ms"] for c in kv[1]))
            for (op, sig), cs in rows[:top]:
                ms = sum(c["ms"] for c in cs)
                lines.append(
                    f"{op[:48]:48s} {len(cs):5d} {ms:8.2f} {ms / len(cs):8.3f}  {sig}"
                )
            return "\n".join(lines)

        def ratio(cs):
            vals = [_excess(c) for c in cs if "tt" in c]
            return max(vals) if vals else 0.0

        rows = sorted(groups.items(), key=lambda kv: -ratio(kv[1]))
        lines.append("tt / sim: worst call's tt mean abs err over the simulation's")
        lines.append("tt and sim against fp64")
        for (op, sig), cs in rows[:top]:
            errs = [c.get("error") for c in cs if "error" in c]
            checked = [c for c in cs if "tt" in c]
            if not checked:
                lines.append(
                    f"{op[:48]:48s} x{len(cs)} not simulated: {errs[:1]}  {sig}"
                )
                continue
            worst = max(checked, key=_excess)
            nonfinite = sum(c.get("nonfinite", 0) for c in checked)
            head = f"{op[:48]:48s} x{len(cs)} tt/sim {ratio(cs):8.2f}"
            lines.append(f"{head} nonfinite {nonfinite}  {sig}")
            lines.append(f"    tt  {format_stats(worst['tt'])}")
            lines.append(f"    sim {format_stats(worst['sim'])}")
        return "\n".join(lines)
