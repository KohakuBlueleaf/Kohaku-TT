"""OpInfo coverage harness: PyTorch's sample inputs per op, run on tt vs fp64 CPU.

Each sample ends as "ok" (with error statistics), "unsupported" (the contract's
KohakuTTUnsupportedOperator) or "error" (anything else: a bug to fix).
    python -m kohakutt.tools.coverage [--ops add mm] [--out <stats.json>]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_map

import kohakutt
from kohakutt.core.stats import diff_stats

# OpInfo names covered by tt kernels, by op family.
TARGETS = [
    # core
    "add",
    "mm",
    "nn.functional.silu",
    "nn.functional.linear",
    "matmul",
    # binary / compare
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "pow",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    # unary
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "tanh",
    "sigmoid",
    "nn.functional.relu",
    "nn.functional.gelu",
    "sin",
    "cos",
    # reductions / nn / optimizer ternaries
    "sum",
    "mean",
    "amax",
    "softmax",
    "log_softmax",
    "nn.functional.cross_entropy",
    "addcmul",
    "addcdiv",
    "lerp",
    # HF Llama / Qwen2
    "index_select",
    "argmax",
    "cumsum",
    "tril",
    "isin",
    "any",
    "all",
    "bitwise_and",
    "bitwise_or",
    "bitwise_not",
    "nn.functional.embedding",
    "nn.functional.layer_norm",
    "bmm",
    "cat",
    "clamp",
    "erf",
    # unary family 2
    "acos",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "acosh",
    "cosh",
    "sinh",
    "tan",
    "exp2",
    "expm1",
    "log2",
    "log10",
    "log1p",
    "erfc",
    "erfinv",
    "lgamma",
    "digamma",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "frac",
    "nn.functional.hardsigmoid",
    "nn.functional.hardswish",
    "nn.functional.mish",
    # composed kernels
    "addmm",
    "amin",
    "argmin",
    "max.reduction_with_dim",
    "min.reduction_with_dim",
    "nn.functional.leaky_relu",
    "nn.functional.elu",
    "nn.functional.hardtanh",
    "isnan",
    "isinf",
    "logical_and",
    "logical_or",
    "logical_xor",
    "remainder",
    "fmod",
    "var",
    "nn.functional.pad.constant",
    "repeat",
    "gather",
    # Core ATen
    "atan2",
    "prod",
    "flip",
    "nonzero",
    "nn.functional.batch_norm",
    "nn.functional.adaptive_avg_pool2d",
    # composite audit
    "select_scatter",
    "slice_scatter",
    "round",
    "round.decimals_3",
    "round.decimals_neg_3",
    "native_group_norm",
    "nn.functional.conv1d",
    "nn.functional.conv2d",
    "sort",
    "topk",
    # strided reads, scatter family
    "diagonal",
    "scatter",
    "scatter_add",
    "scatter_reduce.sum",
    "scatter_reduce.prod",
    "scatter_reduce.mean",
    "index_put",
    "index_fill",
    "masked_scatter",
    # pooling
    "nn.functional.max_pool2d",
    "max_pool2d_with_indices_backward",
    "nn.functional.avg_pool2d",
    "nn.functional.avg_pool1d",
    "nn.functional.adaptive_avg_pool1d",
    # padding modes, transposed conv
    "nn.functional.pad.reflect",
    "nn.functional.pad.replicate",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    # remaining Core ATen
    "cdist",
    "nn.functional.pdist",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.unfold",
    "nn.functional.embedding_bag",
    "grid_sampler_2d",
    "nn.functional.max_pool3d",
    # sweep ranking: linalg / pointwise / stats / shape
    "dot",
    "mv",
    "addmv",
    "baddbmm",
    "addbmm",
    "norm",
    "dist",
    "rsub",
    "div.floor_rounding",
    "div.trunc_rounding",
    "floor_divide",
    "std",
    "std_mean",
    "logaddexp",
    "logsumexp",
    "xlogy",
    "fmax",
    "fmin",
    "copysign",
    "nn.functional.logsigmoid",
    "nn.functional.binary_cross_entropy_with_logits",
    "aminmax",
    "count_nonzero",
    "equal",
    "roll",
    "unfold",
    "cumprod",
    "nn.functional.kl_div",
    "nn.functional.nll_loss",
    # upsampling, non-finite predicates
    "nn.functional.interpolate.nearest",
    "nn.functional.interpolate.nearest-exact",
    "nn.functional.interpolate.linear",
    "nn.functional.interpolate.bilinear",
    "nn.functional.interpolate.trilinear",
    "isfinite",
    "isposinf",
    "isneginf",
    "nan_to_num",
    # sweep tail: models / utilities
    "index_add",
    "vdot",
    "linalg.cross",
    "heaviside",
    "angle",
    "bucketize",
    "searchsorted",
    "nonzero_static",
    "nn.functional.adaptive_max_pool2d",
    "nn.functional.adaptive_max_pool1d",
    "_batch_norm_with_update",
    "nn.functional.max_unpool1d",
    "nn.functional.max_unpool2d",
    "nn.functional.max_unpool3d",
    "index_reduce.prod",
    "index_reduce.mean",
    # scans, order statistics, histograms
    "cummax",
    "cummin",
    "median",
    "nanmedian",
    "histc",
    "i0",
    # losses, activations, selection utilities
    "nn.functional.mse_loss",
    "nn.functional.binary_cross_entropy",
    "nn.functional.huber_loss",
    "nn.functional.smooth_l1_loss",
    "nn.functional.softplus",
    "nn.functional.glu",
    "nn.functional.hardshrink",
    "nn.functional.softshrink",
    "nn.functional.threshold",
    "nn.functional.prelu",
    "logit",
    "nansum",
    "sinc",
    "sgn",
    "signbit",
    "logaddexp2",
    "lerp",
    "repeat_interleave",
    "masked_select",
    "take",
    "trace",
    "kthvalue",
    # N-d pools, margin losses, special functions, utilities
    "nn.functional.avg_pool2d",
    "nn.functional.avg_pool3d",
    "nn.functional.adaptive_max_pool3d",
    "nn.functional.multi_margin_loss",
    "nn.functional.multilabel_margin_loss",
    "nn.functional.rrelu",
    "nn.functional.channel_shuffle",
    "nn.functional.cosine_similarity",
    "hypot",
    "renorm",
    "logcumsumexp",
    "mode",
    "nanmean",
    "unique",
    "unique_consecutive",
    "put",
    "index_copy",
    "digamma",
    "mvlgamma.mvlgamma_p_1",
    "polygamma.polygamma_n_0",
    "polygamma.polygamma_n_1",
    "polygamma.polygamma_n_2",
    "polygamma.polygamma_n_3",
    "polygamma.polygamma_n_4",
    "special.entr",
    "special.xlog1py",
    "special.log_ndtr",
    "special.ndtri",
    "special.i0e",
    "special.i1",
    "special.i1e",
    "special.erfcx",
    # FFT (fp32 / complex64 samples)
    *(
        f"fft.{name}"
        for name in (
            "fft",
            "ifft",
            "rfft",
            "irfft",
            "hfft",
            "ihfft",
            "fft2",
            "ifft2",
            "rfft2",
            "irfft2",
            "hfft2",
            "ihfft2",
            "fftn",
            "ifftn",
            "rfftn",
            "irfftn",
            "hfftn",
            "ihfftn",
            "fftshift",
            "ifftshift",
        )
    ),
]

# OpInfo by name, and by name.variant for variants (e.g. max.reduction_with_dim).
OPINFOS = {}
for _op in op_db:
    if _op.variant_test_name:
        OPINFOS[f"{_op.name}.{_op.variant_test_name}"] = _op
        OPINFOS.setdefault(_op.name, _op)
    else:
        OPINFOS[_op.name] = _op


def _move(value, device, dtype=None):
    """value with every tensor moved to `device`; floating ones cast to `dtype`
    (complex ones to complex128 when `dtype` is float64)."""

    def move(x):
        if not isinstance(x, torch.Tensor):
            return x
        cast = x.dtype
        if dtype is not None and x.is_floating_point():
            cast = dtype
        elif dtype == torch.float64 and x.is_complex():
            cast = torch.complex128
        return x.to(device=device, dtype=cast)

    return tree_map(move, value)


def _outcome(error):
    """unsupported / error for an exception raised on tt: the contract's refusal, or
    PyTorch's own "Could not run ... backend" (e.g. sparse layouts), is unsupported."""
    message = str(error)
    if isinstance(error, NotImplementedError) and (
        kohakutt.UNSUPPORTED in message or message.startswith("Could not run")
    ):
        return "unsupported"
    return "error"


def run_sample(op, sample):
    """One record for `sample` of OpInfo `op` (bf16 inputs)."""
    args = (sample.input, *sample.args)
    shapes = [list(a.shape) for a in args if isinstance(a, torch.Tensor)]
    record = {"shapes": shapes, "kwargs": sorted(sample.kwargs)}
    try:
        tt_args, tt_kwargs = _move(args, "tt"), _move(sample.kwargs, "tt")
        result = op.op(*tt_args, **tt_kwargs)
        got = _move(result, "cpu")
    except Exception as error:  # noqa: BLE001  (classified, never swallowed silently)
        record["outcome"] = _outcome(error)
        record["message"] = str(error).splitlines()[0][:200]
        # A sample PyTorch itself rejects on CPU (same bf16 inputs) is a harness case.
        if record["outcome"] == "error" and "to be on cpu" in record["message"]:
            record["outcome"] = "harness"  # the op requires that argument on CPU
        if record["outcome"] == "error":
            try:
                op.op(*args, **sample.kwargs)
            except Exception:  # noqa: BLE001  (CPU rejects it too)
                record["outcome"] = "harness"
        return record

    # fp64 reference on the same bf16 inputs; a failing reference is a harness issue.
    try:
        ref = op.op(
            *_move(args, "cpu", torch.float64),
            **_move(sample.kwargs, "cpu", torch.float64),
        )
    except Exception as error:  # noqa: BLE001  (classified as a harness outcome)
        record.update(outcome="harness", message=str(error).splitlines()[0][:200])
        return record
    got_t = got[0] if isinstance(got, tuple | list) and got else got
    ref_t = ref[0] if isinstance(ref, tuple | list) and ref else ref
    if not isinstance(ref_t, torch.Tensor) or not isinstance(got_t, torch.Tensor):
        # Python results (bool, number, ...): exact equality.
        same = bool(got_t == ref_t)
        record["outcome"] = "ok" if same else "error"
        if not same:
            record["message"] = f"result {got_t!r} != {ref_t!r}"[:200]
        return record
    if tuple(got_t.shape) != tuple(ref_t.shape):
        record.update(
            outcome="error", message=f"shape {list(got_t.shape)} != {list(ref_t.shape)}"
        )
        return record
    if ref_t.numel() == 0:
        record["outcome"] = "ok"
        return record
    note = _unmeasurable(op, sample)
    if note:
        record.update(outcome="ok", note=note)
        return record
    if got_t.is_complex() or ref_t.is_complex():
        got_t, ref_t = torch.view_as_real(got_t), torch.view_as_real(
            ref_t.to(got_t.dtype)
        )
    try:
        stats = diff_stats(got_t, ref_t)
    except Exception as error:  # noqa: BLE001  (classified as a harness outcome)
        record.update(outcome="harness", message=str(error).splitlines()[0][:200])
        return record
    record.update(outcome="ok", stats=stats, cpu_stats=_cpu_stats(op, sample, ref_t))
    return record


_UNINITIALIZED = ("empty", "new_empty", "resize_", "resize_as_")
_RANDOM = (
    "bernoulli",
    "normal",
    "uniform",
    "rand",
    "multinomial",
    "exponential",
    "geometric",
    "cauchy",
    "log_normal",
    "nn.functional.dropout",
    "nn.functional.alpha_dropout",
    "nn.functional.feature_alpha_dropout",
    "nn.functional.rrelu",
)


def _unmeasurable(op, sample):
    """'uninitialized' or 'random' when the sample's values are not comparable with
    the CPU's (shape checked only), else None."""
    if op.name.startswith(_UNINITIALIZED):
        return "uninitialized"
    kwargs = sample.kwargs
    if (
        op.name.startswith(_RANDOM)
        or (kwargs.get("dropout_p") or 0) > 0
        or kwargs.get("training") is True
    ):
        return "random"
    return None


def _cpu_stats(op, sample, ref_t):
    """diff_stats of the CPU result at the sample's dtype against `ref_t`; None when
    the CPU run fails or its first result is not a tensor of `ref_t`'s shape."""
    try:
        cpu = op.op(sample.input, *sample.args, **sample.kwargs)
        cpu_t = cpu[0] if isinstance(cpu, tuple | list) and cpu else cpu
        if isinstance(cpu_t, torch.Tensor) and cpu_t.is_complex():
            cpu_t = torch.view_as_real(cpu_t)
        if not isinstance(cpu_t, torch.Tensor) or cpu_t.shape != ref_t.shape:
            return None
        return diff_stats(cpu_t, ref_t)
    except Exception:  # noqa: BLE001  (no CPU baseline for this sample)
        return None


def sample_dtypes(op):
    """bf16 when `op` supports it on CPU, else fp32, else int32 / int64 / bool (ops with
    no floating dtype); plus complex64 for fft.* ops (the only complex scope on tt)."""
    supported = op.supported_dtypes("cpu")
    order = (torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool)
    dtypes = [next((d for d in order if d in supported), torch.float32)]
    if op.name.startswith("fft.") and torch.complex64 in supported:
        dtypes.append(torch.complex64)
    return dtypes


def run_op(name, max_samples=24):
    """Summary + per-sample records for OpInfo `name`, per sample_dtypes(op)."""
    op = OPINFOS[name]
    samples = []
    for dtype in sample_dtypes(op):
        try:
            made = list(op.sample_inputs("cpu", dtype))[:max_samples]
        except Exception as error:  # noqa: BLE001  (OpInfo cannot make these samples)
            made = []
            reason = str(error).splitlines()[0][:200] or type(error).__name__
            print(f"[coverage] {name}: no {dtype} samples ({reason})", file=sys.stderr)
        samples += [(dtype, sample) for sample in made]
    records = []
    for index, (dtype, sample) in enumerate(samples):
        # Progress on stderr, so a crash inside a kernel names its op and sample.
        print(f"[coverage] {name} #{index}", file=sys.stderr, flush=True)
        records.append({"dtype": str(dtype).split(".")[-1], **run_sample(op, sample)})
    counts = {
        k: sum(r["outcome"] == k for r in records)
        for k in ("ok", "unsupported", "error", "harness")
    }
    measured = [r for r in records if "stats" in r]
    return {
        "op": name,
        **counts,
        "max_abs": max((r["stats"]["abs_max"] for r in measured), default=None),
        "min_cosine": min((r["stats"]["cosine"] for r in measured), default=None),
        "samples": records,
    }


def main():
    """Runs the harness on the open device 0 and writes the stats file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+", default=TARGETS)
    parser.add_argument(
        "--all", action="store_true", help="every OpInfo entry (name.variant keys)"
    )
    parser.add_argument("--start", default=None, help="resume --all from this op")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--out", default="docs/reference/coverage_stats.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="1x2 mesh of chips 0, 1 with every input sharded where a dim divides",
    )
    args = parser.parse_args()

    names = args.ops
    if args.all:
        names = sorted({f"{o.name}.{o.variant_test_name}".rstrip(".") for o in op_db})
        if args.start:
            names = names[names.index(args.start) :]
    if args.mesh:
        os.environ["KTT_MESH_AUTO"] = "1"
        os.environ["KTT_MESH_SHARD_MIN"] = "0"
        kohakutt.mesh([0, 1], shape=(1, 2))
        torch.ops.ktt.reset_mesh_counters()
    else:
        kohakutt.open(args.device)
    try:
        results = [run_op(name, args.max_samples) for name in names]
        if args.mesh:
            gathers, nbytes, device_gathers, failures = torch.ops.ktt.mesh_counters()
            print(
                f"mesh: {gathers} gather fallbacks ({nbytes} B), "
                f"{device_gathers} through collectives, {failures} collective failures"
            )
    finally:
        kohakutt.close()
    for r in results:
        print(
            f"{r['op']:28s} ok {r['ok']:3d}  unsupported {r['unsupported']:3d}  "
            f"error {r['error']:3d}  max abs {r['max_abs']}  "
            f"min cosine {r['min_cosine']}"
        )

    # Error samples are bugs: list each one.
    for r in results:
        for s in r["samples"]:
            if s["outcome"] == "error":
                print(f"  ERROR {r['op']} {s['shapes']}: {s['message'][:160]}")
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
