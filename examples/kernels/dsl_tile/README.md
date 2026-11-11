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
