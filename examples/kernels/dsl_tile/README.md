# Kernels in the DSL's tile level

Kernels written with `import kohakutt.dsl as kd` (the guide:
[docs/concepts/dsl-tile.md](../../../docs/concepts/dsl-tile.md)): each says what one unit
of work loads, computes and stores, and is lowered to a metal-level program.

| file | kernels | compared with |
|---|---|---|
| `swiglu.py` | `silu`, `swiglu` (tiles) | the metal-level SiLU and `../dsl_metal/swiglu_km.py` (bit for bit), `F.silu(a) * b` eagerly |
| `rms_norm.py` | `rms_norm` (rows, two passes, row held in L1), `rms_norm_split` (a row split over a group of cores) | `../dsl_metal/rmsnorm_km.py` (bit for bit), Gemma's norm module eagerly |
| `layer_norm.py` | `layer_norm` (rows with column values: exact mean, then the exact variance about it; rows split over core groups from 4096 columns) | float64, `F.layer_norm` |
| `group_norm.py` | `group_norm` (NCHW: one run of tiles per (sample, group), exact mean and variance over the run, per-channel affine) | float64, `F.group_norm`, at the SDXL UNet's and VAE's shapes |
| `rms_norm_backward.py` | `rms_norm_backward` (dx and the per-row rstd), `rms_norm_dweight` (strided column runs, exact sums over rows) | float64 torch autograd, the same formulas as eager ops |
| `rope.py` | `rope_half` (rows, `kd.cat` of slices), `rope_complex` (tiles, column pairs) | `../dsl_metal/rope_km.py` (complex, bit for bit), the DINOv3 / Qwen and PixAI tagger code eagerly |
| `online_softmax.py` | `softmax_step` (rows with column values: matrix-unit max, exact sum of the exps, running max and sum in and out) | the float64 step, the same step in eager torch ops, `torch.softmax` |
| `linear.py` | `linear`, `linear_silu`, `linear_gelu` (matmul + bias + GELU), `norm_linear` (RMS norm as the matmul's prologue) | `../dsl_metal/matmul_km.py` (bit for bit), `linear(rms_norm(x))` (bit for bit), eager ops |
| `shared_linear.py` | `linear`, `linear_gelu` with `share=g.col` (w multicast down core columns); `linear_2d`, `linear_gelu_2d` with a and w shared (2-D) | `../dsl_metal/gemm_km.py` and the backend's `a @ w` (bit for bit where it runs its block matmul), eager GELU |
| `flash_attention.py` | `flash_attention` (carried running max and sum, the accumulator; causal, grouped heads, ragged lengths) | float64, the backend's flash kernel (equal numerics: its default exponential) |
| `gdn_decode.py` | `gdn_decode` (a gated delta net's decode step: state decay, delta-rule update, output; q, k, v and the gates laid out by the host) | float64, the same step as eager torch ops |
| `in_use.py` | `RMSNormFunction` (an autograd Function over `rms_norm` and the backward pair), a small MLP block with tile kernels | float64 autograd, eager calls inside `kohakutt.capture` and `torch.compile(backend="ktt_tt")`, the same model with torch-op norms (training losses) |

## Measured

Blackhole P150A, one card, us per call (best of 7 rounds of 20 calls, the timing helper
of `common.py`); "memory bound" is the bytes moved at a nominal 512 GB/s (320-357 GB/s
measured). Errors are against float64 unless named. Host load average 18 to 24 during
the runs (a shared host); single shapes vary by up to 2x between runs, the device
spans quoted below do not.

| kernel | shape | tile | eager or backend | memory bound | note |
|---|---|---|---|---|---|
| `rms_norm` | [1, 2048, 1536] | 70 | `F.rms_norm` 97, Gemma module 579 | 25 | `group=1`: bit for bit with the metal-level norm |
| `rms_norm` | [1, 4096, 1024] | 61 | `F.rms_norm` 78 | 33 | each row split over 4 cores |
| `rms_norm` | [1, 4096, 4096] | 197 | `F.rms_norm` 226 | 131 | |
| `rms_norm` | [1, 4096, 8, 256] | 405 | `F.rms_norm` 329, Gemma module 5136 | 66 | device span equal (94 against 94 us) |
| `rms_norm_split` | [1, 128, 8192], group 16 | 81 | one core per row 261 | | |
| `swiglu` | [1, 4096, 4096] | 250 | metal level 235, `F.silu(a) * b` 417 | 197 | |
| `rope_half` | [1, 16, 1024, 64] | 51 | 380 | 9 | |
| `rope_complex` | [1, 16, 5184, 64] | 106 | metal level 105, eager 278 | 44 | |
| `softmax_step` | [1, 16, 4096, 1024] | 1050 | `torch.softmax` 903, eager step 7818 | 526 | sum within 2e-6 |
| `layer_norm` | [1, 2048, 1024] | 60 | `F.layer_norm` 84 | 16 | bf16 error equal to eager's |
| `layer_norm` | [1, 4096, 1024] | 92 | 87 | 33 | slower on the device (below) |
| `layer_norm` | [1, 4096, 4096] | 253 | 247 | 131 | slower on the device (below) |
| `group_norm` | [2, 320, 64, 64] | 104 | `F.group_norm` 1372 | 20 | error at most eager's |
| `group_norm` | [1, 512, 128, 128] | 507 | 1115 | 66 | |
| `rms_norm_backward` + `rms_norm_dweight` | [2048, 1536] | 298 | the same formulas as eager ops 1375 | | rel. error 2.2e-6 / 2.4e-6 |
| `linear_2d` | [2048, 768] @ [768, 3072] | 98 | backend `a @ w` 103 | | bit for bit |
| `linear_gelu_2d` | [4096, 1024] @ [1024, 4096] | 440 | eager matmul + GELU 735 | | |
| `flash_attention` | [1, 14, 2048, 64] | 1043 | backend flash 386 | | rel. RMS 6.6e-3 (backend 6.4e-3) |
| `flash_attention` causal | [1, 14, 2048, 64] | 947 | backend flash 313 | | 5.5e-3 (5.0e-3) |
| `flash_attention` | [1, 24, 4396, 128] | 8083 | backend flash 2890 | | 6.6e-3 (6.4e-3) |
| `gdn_decode` | 1 sequence, 16 / 48 heads, 128 x 128 state | kernel 58 plus 3 programs of about 10.5 each (device, one run); step with its layouts 250 (device, 41 programs) | eager step 900 (device, 62 programs) | 25 | rel. error 1.8e-7 (eager 1.0e-7) |

The norms split each row over a group of cores where it pays (the plan picks the
group), and run their stores as one function over two tiles with each broadcast value
loaded once per band of rows, as the backend's norm does. RMSNorm's device spans are
at or below the backend's fused norm at every measured shape; LayerNorm's are 1.06x
to 1.18x above at most shapes (faster at 2048 x 4096 and with 256 columns). The attention
kernel runs the backend flash kernel's exponential and is at its accuracy, 2.7x to 3x
slower: per score tile its steps spend 1.9 us (device zones) against 0.63, mostly in
vector work on the math thread (the running maximum and its rescale computed over whole
tiles, where the backend keeps them per row in the vector unit's lanes) and in the
accumulator's round trip through the registers every step; with 3 key tiles per block (`keys=3`, one register scope for scores, row
max and p) it is 1.2x to 1.35x slower again (1307 / 1207 / 10936 us).

```bash
python examples/kernels/dsl_tile/rms_norm.py --tokens 2048 --width 1536
python examples/kernels/dsl_tile/flash_attention.py --shape 1 14 2048 64 --causal
python examples/kernels/dsl_tile/in_use.py
python examples/kernels/dsl_tile/bench.py rms     # rms | swiglu | rope | softmax | ln | gn
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
