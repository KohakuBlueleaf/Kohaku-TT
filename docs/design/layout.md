# Layout experiments: TILE vs ROW_MAJOR, conversions and views

Measured on one P150A (fast dispatch, 1x1) with plain TT-NN calls from Python, bf16
n x n, device-synced mean of 20 calls after warm-up.

| n | tilize | untilize | mm (TILE) | tilize x2 + mm | transpose TILE / RM | slice TILE / RM | reshape TILE / RM |
|---|---|---|---|---|---|---|---|
| 256 | 52.6 | 49.5 | 62.4 | 155.1 | 62.1 / 60.9 | 59.6 / 58.1 | 61.0 / 48.4 |
| 1024 | 48.0 | 40.3 | 72.3 | 158.5 | 62.9 / 76.6 | 68.3 / 58.5 | 60.0 / 48.8 |
| 4096 | 223.1 | 181.8 | 3032.6 | 3456.8 | 179.6 / 1194.4 | 117.1 / 103.0 | 718.0 / 184.3 |

All times in µs. Reshape is (n, n) -> (2n, n/2); slice keeps the first n/2 rows.

## Reading

- **Up to 1024 every op costs one dispatch (40-75 µs).** The layout itself is invisible;
  what costs is the number of ops. Matmul from ROW_MAJOR is 2.2x matmul on TILE only
  because it is three dispatches.
- **At 4096 the layouts pull in opposite directions.** Transpose is 6.6x cheaper on
  TILE (in-tile transpose plus tile reorder); reshape that changes the last dim is 3.9x
  cheaper on ROW_MAJOR (TILE must retile). Row slices cost the same.
- **Conversion is bandwidth-cheap.** Tilize of 32 MB: 223 µs (~290 GB/s read+write).
  Tilizing both matmul inputs adds 14% at n=4096 (425 µs on 3.03 ms).
- **Tilize-on-read matmul bound.** A reader that tilizes while loading would save at
  most the two tilize ops: ~14% at 4096, and two dispatches (~100 µs) at small sizes;
  it pays mainly through the dispatch count.

## Direction for tt tensor views

Neither layout wins outright, so storing one layout and materializing every view eagerly
is the wrong default. Direction:

1. tt tensors carry torch strides as metadata over a TILE base; views are lazy.
2. Consumers fold what they can: transpose into matmul (`transpose_a/b` cost nothing),
   row slices into reader offsets.
3. Anything else materializes once, choosing the layout by the op (reshape via
   ROW_MAJOR, transpose via TILE), and the result is cached on the view.

This matches how CUDA keeps strided tensors and lets kernels (cuBLAS `op(A)`) absorb
transposes, with the tile constraint handled at materialization.
