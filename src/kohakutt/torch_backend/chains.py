"""Graph pass of the tt compile backends (Dynamo and AOT graphs): RMSNorm chains become
ktt.rms_norm, rotate-half RoPE ktt_fusion.rope; a bool attention mask is converted once,
repeated K / V heads go to SDPA ungrouped. KTT_FUSE_CHAINS: 0, 1 (all, the default), or
a comma list of norm, rotary, mask, gqa. A failing rewrite leaves the graph."""

import contextlib
import copy
import operator
import os
import warnings

import torch
import torch.fx
import torch.nn.functional as F

aten = torch.ops.aten

# Canonical op names of torch functions, aten overloads and tensor methods.
FUNCTIONS = {
    operator.mul: "mul",
    torch.mul: "mul",
    aten.mul.Tensor: "mul",
    operator.add: "add",
    torch.add: "add",
    aten.add.Tensor: "add",
    operator.neg: "neg",
    torch.neg: "neg",
    aten.neg.default: "neg",
    torch.rsqrt: "rsqrt",
    aten.rsqrt.default: "rsqrt",
    operator.pow: "pow",
    torch.pow: "pow",
    aten.pow.Tensor_Scalar: "pow",
    torch.mean: "mean",
    aten.mean.dim: "mean",
    torch.cat: "cat",
    aten.cat.default: "cat",
    aten._to_copy.default: "cast",
    torch.ops.prims.convert_element_type.default: "cast",
    operator.getitem: "getitem",
    aten.slice.Tensor: "slice",
    torch.reshape: "reshape",
    aten.reshape.default: "reshape",
    aten.view.default: "reshape",
    aten._unsafe_view.default: "reshape",
    aten.clone.default: "clone",
    aten.expand.default: "expand",
    aten.unsqueeze.default: "unsqueeze",
}
METHODS = {
    "mul": "mul",
    "add": "add",
    "neg": "neg",
    "__neg__": "neg",
    "rsqrt": "rsqrt",
    "pow": "pow",
    "mean": "mean",
    "to": "cast",
    "float": "cast",
    "type_as": "cast",
    "bfloat16": "cast",
    "reshape": "reshape",
    "view": "reshape",
    "contiguous": "clone",
    "expand": "expand",
    "unsqueeze": "unsqueeze",
}

REWRITES = ("norm", "rotary", "mask", "gqa")
DEFAULT = REWRITES


def chosen():
    """The rewrites KTT_FUSE_CHAINS (read per compile) selects: 0 / off / false none,
    1 / on / true (or unset) the default, `all` every one, or a comma list of names
    (case and spaces ignored). An unknown name raises ValueError."""
    raw = os.environ.get("KTT_FUSE_CHAINS")
    tokens = [t.strip().lower() for t in (raw or "").split(",") if t.strip()]
    if not tokens or tokens in (["1"], ["on"], ["true"]):
        return set(DEFAULT)
    if tokens in (["0"], ["off"], ["false"]):
        return set()
    unknown = [t for t in tokens if t not in REWRITES and t != "all"]
    if unknown:
        raise ValueError(
            f"KTT_FUSE_CHAINS={raw!r}: unknown {', '.join(unknown)}; use 0, 1, all or "
            f"a comma list of {', '.join(REWRITES)}"
        )
    return set(REWRITES) if "all" in tokens else set(tokens)


def enabled(rewrite):
    """True when KTT_FUSE_CHAINS lets `rewrite` run."""
    return rewrite in chosen()


def _kind(n):
    if not isinstance(n, torch.fx.Node):
        return None
    if n.op == "call_function":
        return FUNCTIONS.get(n.target)
    if n.op == "call_method":
        return METHODS.get(n.target)
    return None


def _first(n):
    """The node's first tensor operand (positional or keyword `input` / `self`)."""
    if n.args:
        return n.args[0]
    return n.kwargs.get("input", n.kwargs.get("self"))


def _val(n):
    """The fake tensor a node computes (Dynamo: example_value, AOT: val), or None."""
    if not isinstance(n, torch.fx.Node):
        return None
    v = n.meta.get("val", n.meta.get("example_value"))
    return v if isinstance(v, torch.Tensor) else None


def _concrete(*vals):
    """True when every value is a tensor of concrete (not symbolic) sizes."""
    return all(v is not None and all(isinstance(d, int) for d in v.shape) for v in vals)


def _meta_like(template, value):
    """Meta for a new node computing `value`: `template`'s meta with its value key."""
    meta = dict(template.meta)
    meta["val" if "val" in template.meta else "example_value"] = value
    return meta


def _fake(v, dtype):
    """A fake tensor like `v` (its mode, device, sizes) of `dtype`."""
    mode = getattr(v, "fake_mode", None)
    with mode if mode is not None else contextlib.nullcontext():
        return torch.empty_like(v, dtype=dtype)


def _arg(n, i, name, default=None):
    if len(n.args) > i:
        return n.args[i]
    return n.kwargs.get(name, default)


def _set(n, i, name, value):
    """Sets argument `i` (keyword `name` when not given positionally) of node n."""
    if len(n.args) > i:
        args = list(n.args)
        args[i] = value
        n.args = tuple(args)
    else:
        n.kwargs = {**n.kwargs, name: value}


def _is_cast(n):
    """True when n only changes its input's floating dtype (same shape and device)."""
    if _kind(n) != "cast" or _first(n) is None:
        return False
    a, b = _val(_first(n)), _val(n)
    return (
        a is not None
        and b is not None
        and a.shape == b.shape
        and a.device == b.device
        and a.dtype.is_floating_point
        and b.dtype.is_floating_point
    )


def _binary(n, kind):
    """(a, b) of a two-operand node of `kind` without alpha, else None."""
    if _kind(n) != kind or len(n.args) != 2 or n.kwargs.get("alpha", 1) != 1:
        return None
    return n.args


def _other(pair, node):
    """The operand of `pair` that is not `node` (None when node is not in it)."""
    a, b = pair
    return b if a is node else a if b is node else None


def _last_dim(dim, rank):
    dims = dim if isinstance(dim, (list, tuple)) else [dim]
    return len(dims) == 1 and isinstance(dims[0], int) and dims[0] % rank == rank - 1


def _erase(graph, nodes):
    """Erases the nodes of `nodes` left without users, last first."""
    for n in reversed(list(graph.nodes)):
        if n in nodes and not n.users:
            graph.erase_node(n)


# ------------------------------------------------------------ in-place updates
def _in_place(n):
    """True when node n writes a tensor in place (a trailing-underscore method or
    function, setitem, or a mutable aten overload)."""
    if n.op not in ("call_function", "call_method"):
        return False
    if n.target is operator.setitem:
        return True
    schema = getattr(n.target, "_schema", None)
    if schema is not None:
        return schema.is_mutable
    name = n.target if n.op == "call_method" else getattr(n.target, "__name__", "")
    return isinstance(name, str) and name.endswith("_") and not name.endswith("__")


def _root(n):
    """The node n derives from through first operands (a placeholder or a creating op):
    every view of a tensor, and more, shares its root."""
    seen = set()
    while (
        isinstance(n, torch.fx.Node)
        and n.op in ("call_function", "call_method")
        and n not in seen
    ):
        seen.add(n)
        first = _first(n)
        if not isinstance(first, torch.fx.Node):
            break
        n = first
    return n


def _written_between(start, end, tensors):
    """True when an in-place update of a tensor sharing a root with `tensors` lies after
    node `start` and before node `end` (graph order)."""
    roots = {_root(t) for t in tensors if isinstance(t, torch.fx.Node)}
    n = start.next
    while n is not end and n.op != "root":
        if _in_place(n) and _first(n) is not None and _root(_first(n)) in roots:
            return True
        n = n.next
    return False


def _earliest(graph, nodes):
    """The node of `nodes` that comes first in graph order."""
    for n in graph.nodes:
        if n in nodes:
            return n
    return None


# ---------------------------------------------------------------------- RMSNorm
def _rms_core(r):
    """For rsqrt node r of rsqrt(mean(x^2, -1, keepdim) + eps): (core, x, eps, chain)
    with core = x * r; else None."""
    add = _first(r) if _kind(r) == "rsqrt" else None
    pair = _binary(add, "add")
    if pair is None:
        return None
    mean, eps = (pair[0], pair[1]) if _kind(pair[0]) == "mean" else (pair[1], pair[0])
    if _kind(mean) != "mean" or not isinstance(eps, (int, float)):
        return None
    sq = _first(mean)
    rank = _val(sq).dim() if _val(sq) is not None else 0
    dim = _arg(mean, 1, "dim")
    if rank == 0 or not _last_dim(dim, rank) or not _arg(mean, 2, "keepdim", False):
        return None
    if _arg(mean, 3, "dtype") is not None:
        return None
    square = _binary(sq, "mul") and sq.args[0] is sq.args[1]
    if not square and not (_kind(sq) == "pow" and _arg(sq, 1, "exponent") == 2):
        return None
    x = _first(sq)
    for core in r.users:
        pair = _binary(core, "mul")
        if pair is not None and _other(pair, r) is x:
            return core, x, float(eps), {core, r, add, mean, sq}
    return None


def _rms_norm(graph, r):
    """Rewrites the RMSNorm chain at rsqrt node r; True when it did."""
    found = _rms_core(r)
    if found is None:
        return False
    core, x, eps, chain = found
    if any(len(n.users) != 1 for n in chain - {core}):
        return False
    source = _first(x) if _is_cast(x) else x
    xv = _val(source)
    if not _concrete(xv) or xv.dim() == 0 or not xv.dtype.is_floating_point:
        return False
    width = xv.shape[-1]
    # Forward from x * r: casts and one multiplication by a weight row.
    end, weight = core, None
    while len(end.users) == 1:
        user = next(iter(end.users))
        pair = _binary(user, "mul")
        if _is_cast(user):
            end = user
        elif weight is None and pair is not None:
            w = _other(pair, end)
            wv = _val(w)
            if not _concrete(wv) or tuple(wv.shape) != (width,):
                break
            if not wv.is_floating_point():
                break
            end, weight = user, w
        else:
            break
        chain.add(end)
    ev = _val(end)
    # Dynamo graphs: chains that need a gradient stay (the fused op has no derivative);
    # AOT forward graphs keep saved activations as extra users, which refuses them.
    if not _concrete(ev) or ev.shape != xv.shape or ev.requires_grad:
        return False
    # ktt.rms_norm returns its input's dtype: the chain's result dtype, else no match
    # (a wider result than the source reads the cast to it).
    if ev.dtype != xv.dtype:
        if x is source or _val(x).dtype != ev.dtype:
            return False
        source, xv = x, _val(x)
    # The fused op reads source and weight where the chain ends.
    first = _earliest(graph, chain | {x})
    if _written_between(first, end, (source, weight)):
        return False
    with graph.inserting_before(end):
        out = graph.call_function(
            torch.ops.ktt.rms_norm.default, (source, weight, None, eps)
        )
    out.meta = _meta_like(end, _fake(xv, ev.dtype))
    end.replace_all_uses_with(out)
    if x is not source:
        chain.add(x)
    _erase(graph, chain)
    return True


def _norms(graph):
    return sum(
        1 for n in list(graph.nodes) if _kind(n) == "rsqrt" and _rms_norm(graph, n)
    )


# ----------------------------------------------------------------------- RoPE
def _half(n):
    """(source, first) when n slices the first (True) or second (False) half of its
    source's last dim; else None."""
    v = _val(n)
    if v is None:
        return None
    if _kind(n) == "slice":
        src = _first(n)
        sv = _val(src)
        if sv is None or _arg(n, 4, "step", 1) != 1:
            return None
        if not _last_dim(_arg(n, 1, "dim", 0), sv.dim()):
            return None
        start = _arg(n, 2, "start", 0) or 0
    elif _kind(n) == "getitem" and len(n.args) == 2 and isinstance(n.args[1], tuple):
        src, index = n.args
        sv = _val(src)
        if sv is None or not index or not isinstance(index[-1], slice):
            return None
        lead = index[:-1]
        full = slice(None, None, None)
        if not (lead == (Ellipsis,) or lead == (full,) * (sv.dim() - 1)):
            return None
        if index[-1].step not in (None, 1):
            return None
        start = index[-1].start or 0
    else:
        return None
    if not _concrete(v, sv):
        return None
    d = sv.shape[-1]
    if d % 2 or v.shape[-1] != d // 2 or v.shape[:-1] != sv.shape[:-1]:
        return None
    if start not in (0, d // 2):
        return None
    return src, start == 0


def _rotated(cat):
    """x when cat is cat((-x[..., d/2:], x[..., :d/2]), -1); else None."""
    if _kind(cat) != "cat":
        return None
    parts, dim = _arg(cat, 0, "tensors"), _arg(cat, 1, "dim", 0)
    if not isinstance(parts, (list, tuple)) or len(parts) != 2:
        return None
    neg, x1 = parts
    if _kind(neg) != "neg" or _val(cat) is None:
        return None
    if not _last_dim(dim, _val(cat).dim()):
        return None
    second, first = _half(_first(neg)), _half(x1)
    if second is None or first is None or second[1] or not first[1]:
        return None
    return first[0] if first[0] is second[0] else None


def _rope_op(xv, cv):
    """(op, extra args) computing RoPE of x on its device: on tt the own program when
    it takes these shapes (else None), elsewhere ktt.rotary's composite."""
    if xv.device.type != torch._C._get_privateuse1_backend_name():
        return (torch.ops.ktt.rotary.default,)
    from kohakutt.fusion import rope

    if xv.dtype != torch.bfloat16 or rope.plan(xv, cv) is None:
        return None
    return torch.ops.ktt_fusion.rope.default, 0


def _rotary(graph, add):
    """Rewrites x * cos + rotate_half(x) * sin at add node `add`; True when it did."""
    pair = _binary(add, "add")
    if pair is None:
        return False
    for m1, m2 in (pair, pair[::-1]):
        a, b = _binary(m1, "mul"), _binary(m2, "mul")
        if a is None or b is None:
            continue
        for cat, sin in (b, b[::-1]):
            x = _rotated(cat)
            cos = _other(a, x) if x is not None else None
            if cos is None or not isinstance(sin, torch.fx.Node):
                continue
            xv, cv, sv, ov = _val(x), _val(cos), _val(sin), _val(add)
            if not _concrete(xv, cv, sv, ov) or cv.dim() < 2:
                continue
            same = cv.shape == sv.shape and cv.dtype == sv.dtype
            if not same or cv.shape[-2:] != xv.shape[-2:]:
                continue
            if ov.shape != xv.shape or ov.dtype != xv.dtype or ov.requires_grad:
                continue
            neg = _arg(cat, 0, "tensors")[0]
            chain = {add, m1, m2, cat, *_arg(cat, 0, "tensors"), _first(neg)}
            if any(len(n.users) != 1 for n in chain - {add, m1, m2} - {x}):
                continue
            # The fused op reads x, cos and sin where the sum was.
            if _written_between(_earliest(graph, chain), add, (x, cos, sin)):
                continue
            fused = _rope_op(xv, cv)
            if fused is None:
                continue
            with graph.inserting_before(add):
                out = graph.call_function(fused[0], (x, cos, sin, *fused[1:]))
            out.meta = _meta_like(add, _fake(xv, xv.dtype))
            add.replace_all_uses_with(out)
            _erase(graph, chain)
            return True
    return False


def _rotaries(graph):
    count = 0
    for n in list(graph.nodes):
        erased = getattr(n, "_erased", False)
        if not erased and _kind(n) == "add" and _rotary(graph, n):
            count += 1
    return count


# ------------------------------------------------------------------ attention
SDPA = {torch._C._nn.scaled_dot_product_attention, F.scaled_dot_product_attention}
OVERRIDEABLE = aten._scaled_dot_product_fused_attention_overrideable.default
# Positions and keyword names of the SDPA operands (torch-level and the aten op).
SDPA_ARGS = {
    "q": (0, "query"),
    "k": (1, "key"),
    "v": (2, "value"),
    "mask": (3, "attn_mask"),
    "dropout": (4, "dropout_p"),
    "causal": (5, "is_causal"),
    "scale": (6, "scale"),
}
OVERRIDEABLE_ARGS = {**SDPA_ARGS, "mask": (3, "attn_bias"), "scale": (99, "scale")}


def _sdpa(n):
    """{operand: (position, name)} of an SDPA node (torch-level or the overrideable
    aten op), else None."""
    if n.op != "call_function":
        return None
    if n.target in SDPA:
        return SDPA_ARGS
    return OVERRIDEABLE_ARGS if n.target is OVERRIDEABLE else None


def _get(n, spec, key, default=None):
    i, name = spec[key]
    return _arg(n, i, name, default)


def _put(n, spec, key, value):
    i, name = spec[key]
    _set(n, i, name, value)


def _on_tt(n):
    v = _val(n)
    return v is not None and v.device.type == torch._C._get_privateuse1_backend_name()


def _masks(graph):
    """Bool masks of tt SDPA calls (torch-level: converted per call) converted once per
    mask and query dtype while the mask is not updated in place; aten graphs: duplicate
    conversions merged. Returns the count."""
    done, merged = {}, 0
    for n in list(graph.nodes):
        spec = _sdpa(n)
        if spec is not OVERRIDEABLE_ARGS and spec is not None:
            q, mask = _get(n, spec, "q"), _get(n, spec, "mask")
            mv = _val(mask)
            if not _on_tt(q) or mv is None or mv.dtype != torch.bool:
                continue
            key = (mask, _val(q).dtype)
            if key in done and _written_between(done[key][1], n, (mask,)):
                del done[key]
            if key not in done:
                kw = {"dtype": key[1], "device": mv.device}
                with graph.inserting_before(n):
                    keep = graph.call_function(aten.scalar_tensor.default, (0.0,), kw)
                    drop = graph.call_function(
                        aten.scalar_tensor.default, (float("-inf"),), kw
                    )
                    conv = graph.call_function(aten.where.self, (mask, keep, drop))
                conv.meta = _meta_like(n, _fake(mv, key[1]))
                done[key] = (conv, n)
            _put(n, spec, "mask", done[key][0])
            merged += 1
    # Identical scalar_tensor / where nodes (aten graphs), the first kept while none of
    # their operands is updated in place in between.
    first = {}
    for n in list(graph.nodes):
        if n.op != "call_function" or n.target not in (
            aten.scalar_tensor.default,
            aten.where.self,
        ):
            continue
        key = (n.target, n.args, tuple(sorted(n.kwargs.items(), key=str)))
        try:
            hash(key)
        except TypeError:
            continue
        kept = first.get(key)
        if kept is not None and not _written_between(kept, n, n.all_input_nodes):
            n.replace_all_uses_with(kept)
            graph.erase_node(n)
            merged += n.target is aten.where.self
        else:
            first[key] = n
    return merged


FULL = slice(None, None, None)


def _ungrouped(n, heads):
    """(K, chain) when n is repeat_kv(K) to `heads` heads in its exact form (K[:, :,
    None] or K.unsqueeze(2), expanded on dim 2, an optional contiguous copy, merged into
    dim 1), else None."""
    chain = [n]
    if _kind(n) != "reshape":
        return None
    n = _first(n)
    if _kind(n) == "clone":
        chain.append(n)
        n = _first(n)
    if _kind(n) != "expand":
        return None
    chain.append(n)
    n = _first(n)
    if _kind(n) == "getitem":
        index = n.args[1] if len(n.args) == 2 else None
        if index not in ((FULL, FULL, None), (FULL, FULL, None, FULL, FULL)):
            return None
    elif _kind(n) == "unsqueeze":
        if _arg(n, 1, "dim") != 2:
            return None
    else:
        return None
    chain.append(n)
    k = _first(n)
    src, out, inserted = _val(k), _val(chain[0]), _val(chain[-1])
    expanded = _val(chain[-2])
    if not _concrete(src, out, inserted, expanded) or src.dim() != 4:
        return None
    b, h, s, d = src.shape
    if out.dim() != 4 or heads % h:
        return None
    if tuple(inserted.shape) != (b, h, 1, s, d) or tuple(out.shape) != (b, heads, s, d):
        return None
    if tuple(expanded.shape) != (b, h, heads // h, s, d):
        return None
    return k, chain


def _takes_grouped(q, k, v, mask, dropout, causal, scale):
    """True when tt's device attention kernels take q with ungrouped k / v (and mask):
    ktt.sdpa_refusal on meta tensors of their sizes and dtypes (no allocation)."""

    def meta(t):
        return None if t is None else torch.empty(t.shape, dtype=t.dtype, device="meta")

    why = torch.ops.ktt.sdpa_refusal(
        meta(q), meta(k), meta(v), meta(mask), float(dropout), bool(causal), scale
    )
    return why == ""


def _grouped(graph):
    """SDPA calls on repeat_kv K / V given the ungrouped K / V where tt's SDPA takes
    grouped heads for that shape and mask; returns the count."""
    count = 0
    for n in list(graph.nodes):
        spec = _sdpa(n)
        if spec is None:
            continue
        q, k, v = (_get(n, spec, key) for key in ("q", "k", "v"))
        qv = _val(q)
        if not _on_tt(q) or not _concrete(qv) or qv.dim() != 4:
            continue
        kc, vc = _ungrouped(k, qv.shape[1]), _ungrouped(v, qv.shape[1])
        if kc is None or vc is None:
            continue
        (k0, k_chain), (v0, v_chain) = kc, vc
        if any(_val(t).requires_grad for t in (q, k0, v0)):
            continue
        mask = _get(n, spec, "mask")
        mv = _val(mask)
        if mask is not None and not _concrete(mv):
            continue
        args = [_get(n, spec, key, d) for key, d in (("dropout", 0.0), ("causal", 0))]
        scale = n.kwargs.get("scale", _get(n, spec, "scale"))
        if not _takes_grouped(qv, _val(k0), _val(v0), mv, *args, scale):
            continue
        # SDPA reads K / V themselves where it read their repeats.
        start = _earliest(graph, {*k_chain, *v_chain})
        if _written_between(start, n, (k0, v0)):
            continue
        _put(n, spec, "k", k0)
        _put(n, spec, "v", v0)
        if spec is SDPA_ARGS:
            n.kwargs = {**n.kwargs, "enable_gqa": True}
        _erase(graph, {*k_chain, *v_chain})
        count += 1
    return count


REWRITE_FNS = (
    ("mask", "mask", _masks),
    ("gqa", "gqa", _grouped),
    ("norm", "rms_norm", _norms),
    ("rotary", "rotary", _rotaries),
)


def fuse_chains(gm):
    """gm with the chosen rewrites applied (in place); returns gm. Each rewrite runs on
    a copy of the graph, taken only when it succeeds: an exception inside one warns and
    leaves the graph as it was. gm.ktt_chains counts the rewrites."""
    selected = chosen()
    counts = {"rms_norm": 0, "rotary": 0, "mask": 0, "gqa": 0}
    for name, key, fn in REWRITE_FNS:
        if name not in selected:
            continue
        work = copy.deepcopy(gm.graph)
        try:
            count = fn(work)
            work.lint()
        except Exception as e:  # noqa: BLE001  (a pass never fails a compile)
            warnings.warn(
                f"kohakutt: compile rewrite {name!r} failed and was skipped: {e!r}",
                stacklevel=2,
            )
            continue
        if count:
            gm.graph = work
            counts[key] = count
    if any(counts.values()):
        gm.recompile()
    gm.ktt_chains = counts
    return gm


__all__ = ["chosen", "enabled", "fuse_chains"]
