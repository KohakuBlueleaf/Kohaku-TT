# Kernels in the DSL's tile level

Kernels written with `import kohakutt.dsl as kd` (see
[docs/concepts/dsl-tile.md](../../../docs/concepts/dsl-tile.md)): each says what one unit
of work loads, computes and stores, and is lowered to a metal-level program.

| file | kernels | compared with |
|---|---|---|
| `swiglu.py` | `silu`, `swiglu` (tiles) | the metal-level SiLU and `../dsl_metal/swiglu_km.py` (bit for bit), `F.silu(a) * b` eagerly |
| `rms_norm.py` | `rms_norm` (rows, two passes, row held in L1) | `../dsl_metal/rmsnorm_km.py` (bit for bit), Gemma's norm module eagerly |
| `linear.py` | `linear`, `linear_silu`, `linear_gelu` (matmul + bias + GELU), `norm_linear` (RMS norm as the matmul's prologue) | `../dsl_metal/matmul_km.py` (plain linear, bit for bit), `linear(rms_norm(x))` (norm_linear, bit for bit), F.linear + activation and F.rms_norm + `@` eagerly |
| `rope.py` | `rope_half` (rows, `kd.cat` of slices), `rope_complex` (tiles, column pairs) | `../dsl_metal/rope_km.py` (complex, bit for bit), the DINOv3 / Qwen and PixAI tagger code eagerly |
| `shared_linear.py` | `linear`, `linear_gelu` with `share=g.col` on w (multicast down core columns) | `../dsl_metal/gemm_km.py` (plain linear, bit for bit), the unshared kernels of `linear.py`, `F.linear` + GELU eagerly |
| `group_norm.py` | `group_norm` (NCHW: one run of tiles per (sample, group), exact mean and variance over the run, per-channel affine) | float64, `F.group_norm`, at the SDXL UNet's shapes |
| `rms_norm_backward.py` | `rms_norm_backward` (dx and the per-row rstd), `rms_norm_dweight` (strided column runs, exact sums over rows) | float64 torch autograd, the same formulas as eager ops |
| `layer_norm.py` | `layer_norm` (rows with column values: exact mean, then the exact variance about it; the affine chain in two stages) | float64, `F.layer_norm` |
| `in_use.py` | `RMSNormFunction` (a `torch.autograd.Function`: `rms_norm` forward, the backward pair), a small MLP block with tile norms and SwiGLU | float64 autograd (gradients), eager calls (inside `kohakutt.capture` and `torch.compile(backend="ktt_tt")`: equal, nothing built on replay), the same model with torch-op norms (AdamW losses) |
| `online_softmax.py` | `softmax_step` (rows with column values: matrix-unit max, exact sum of the exps, running max and sum in and out) | the float64 step, the same step in eager torch ops, `torch.softmax` (from the empty state) |

```bash
python examples/kernels/dsl_tile/rms_norm.py --tokens 2048 --width 1536
python examples/kernels/dsl_tile/swiglu.py --tokens 2048 --width 4096
python examples/kernels/dsl_tile/rope.py --heads 16 --tokens 1024 --dim 64
python examples/kernels/dsl_tile/online_softmax.py --heads 8 --rows 256 --keys 1024
python examples/kernels/dsl_tile/bench.py rms     # the timing table
python -m pytest examples/kernels/dsl_tile/test_dsl_tile.py   # card tests
```

```python
@kd.kernel
def rms_norm(x: kd.Tensor, scale: kd.Tensor, out: kd.Out, eps: kd.Runtime[float]):
    g = kd.resident(scale.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        r = kd.rsqrt(ms + eps)
        kd.store(out.tiles[t, :], (row * r.col() * g.row()).cast(out.dtype))
```

`rms_norm.plan(x, scale, 1e-6).report()` prints the plan and its cost line;
`rms_norm.metal(x, scale, 1e-6)` the metal-level program it lowers to.
