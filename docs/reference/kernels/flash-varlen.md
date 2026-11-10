---
title: ktt_flash, packed variable-length attention
summary: Attention over sequences of different lengths packed back to back with no padding, forward and backward - the entry points, the layout the kernels read, the per-row plan and its mask tiles, how one program serves every set of lengths, numerics, measured cost, and limits.
tags:
  - reference
  - kernels
  - attention
---

# ktt_flash, packed variable-length attention

`out[Tq, H, Dv]`: every query token attends to the keys of its own sequence.
`q [Tq, H, D]`, `k [Tk, Hk, D]`, `v [Tk, Hk, Dv]` (bf16, H a multiple of Hk) hold the
tokens of all sequences back to back; cumulative offsets `[0, ..., total]` mark the
sequences. Nothing is padded, in or out. The contract is
`torch.nn.attention.varlen.varlen_attn`'s (and `flash_attn_varlen_func`'s).

## Entry points

```python
out = kohakutt.sdpa_varlen(
    q, k, v, cu_seq_q, cu_seq_k=None, is_causal=False, scale=None, attn_mask=None
)
out = torch.nn.attention.varlen.varlen_attn(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k)
```

Both run the same kernel: Kohaku-TT registers a tt kernel for torch's
`torch_attn::_varlen_attn`, and `kohakutt.sdpa_varlen` calls that op. `max_q` / `max_k`
are accepted and not needed. Through torch's function, `window_size` `(-1, -1)` (dense)
and `(-1, 0)` (causal) are taken; `seqused_k`, `block_table` and other windows raise.

- Dense: a query sees all keys of its sequence.
- Causal: query r of a sequence sees its first `min(Lk, r + 1 + Lk - Lq)` keys (the
  diagonal aligned at the last query and key, as Flash; `r + 1` for equal lengths).
- `cu_seq_k` may differ from `cu_seq_q` (other key lengths per sequence).
- Offsets are host values: a list or a CPU int tensor is read once per call. A tt tensor
  is downloaded (a synchronize and a read per call).

Gradients: `dq`, `dk`, `dv` come back packed as `q`, `k`, `v`, through autograd from
either entry point (torch's `_varlen_attn_backward` has a tt kernel). The forward always
writes each query's log-sum-exp (torch's op returns it, `[H, Tq]`, and the backward
reads it); it stays on the device with the output.

## Key mask

`attn_mask` of `kohakutt.sdpa_varlen` is one value per packed key, `[Tk]`, applied for
every query of the key's sequence and every head (torch's `varlen_attn` has no mask
argument):

- bool: True keeps the key; a key left out has -1e4 added to its logits. For every query
  that keeps at least one key it gets P exactly 0 and exact zero gradients.
- float: added to the key's scores, clamped below at -1e4. The value is stored as bf16
  after the division by the scale (about 0.4 % of the value).

No other mask shape is taken (per query, per head, `[Tq, Tk]`). The mask gets no
gradient. The mask is finite, so a query whose visible keys are all left out attends to
them as if they were kept (softmax over their scores; torch gives NaN), and those keys
get its gradients. With `is_causal` that is the first query of a sequence whose first
key is left out.

The mask is a tensor like `k`: a tt mask is converted on the device (to bf16, additive,
divided by the scale) and is never downloaded; a CPU mask is converted on the host and
uploaded. The kernels read it as `[1, Tk]`, one row tile per key tile, streamed with K
and V. A score tile without a plan mask tile takes the row by the dense kernel's row
broadcast; where it has a plan mask tile, the reader writes the row's values into that
tile's visible entries in L1 before the compute copies it. The plan and its cache do not
depend on the mask. Forward and both backward programs take it, with a mask of all True
(or all zero) giving bit-identical results to no mask.

## What the kernels read

A token's heads lie side by side: the kernels read `[T, H D]` and write `[Tq, H Dv]`,
which is the storage of a `[T, H, D]` view of a projection's `[T, H D]` output. Such
views cost nothing; the call is one program. A `[T, H, D]` tensor stored on its own is
first reshaped on the device.

A query tile row is 32 consecutive packed tokens of one head. It can hold the end of one
sequence and the start of the next, or several short sequences. The plan (host,
`ktt_flash/varlen_plan.hpp`) gives every query tile row one record, the same for every
head:

- its key tile window `[kfirst, klast]`: the key tiles any of its 32 tokens sees;
- `a`, `b`: how many leading and trailing tiles of that window take a mask tile. The
  tiles between are seen whole by all 32 tokens and run as in the dense kernel;
- the first page of its `a + b` mask tiles. A mask tile holds 0 where the row's token
  sees the tile's key and the masked value elsewhere, and is copied into the score tile
  before the score matmul accumulates onto it, as the dense kernel's masks are.

Sequence boundaries are arbitrary token offsets; sequences shorter than a tile, several
per tile, and a last tile of fewer than 32 tokens are all the same case. The rows past the
last token in the last tile of q, k and v are zeroed by the reader in L1 after it reads
the tile, so undefined tile padding never reaches a score.

## Plan constants

The records (int32, 8 words per query tile row) and the mask tiles (bf16) are built on
the host from the offsets and uploaded once per (offsets, causal, scale); the 32 most
recent sets stay on the device (the backward's key-side set counts as one). They are constants of the length set, shared by all
heads and by every call with those lengths. Sizes: 8 sequences of 201 tokens take 176
mask tiles (360 KB); 700 + 13 + 1500 tokens take 187 (383 KB); lengths that are
multiples of 32 take none.

The masked value is a logit of -5e5 (`kMaskedLogit` over the scale). A query tile row
can meet key tiles in which some of its tokens see no key (the other sequence's keys);
until their first visible key their running softmax reference is the masked value, and
fp32 has to carry it exactly, which rules out the dense kernel's -1e30. When the first
visible key arrives the reference moves and everything accumulated before is scaled by
exactly 0.

## Backward

Two programs, the dense backward's ([flash.md](flash.md), Backward) with `FLASH_VARLEN`:

- dQ and the row statistics walk query tile rows: the forward's records and mask tiles.
- dK / dV walk key tile rows, so they read the plan from the keys' side: one record per
  key tile row (the query tile rows that see it, the leading and trailing ones taking a
  mask tile) and the same mask tiles in that order. This second record set is built and
  uploaded on the first backward of a set of lengths; a training step keeps both (twice
  the mask tiles of the forward).

A masked score gives P exactly 0, so dS is exactly 0 for the keys of other sequences. The
readers zero the rows past the last token of Q, dO, O, K and V, which makes dS exactly 0
there. The keys of a sequence without queries get exact zero gradients. A sequence's
gradients are bit-identical whatever its neighbours hold. The scores are computed as the
forward computes them (S = Q K^T); the key-row orientation is not used.

## One program for every set of lengths

The kernels are the dense forward's sources with `FLASH_VARLEN`: tile counts, the
jobs' key windows and the heads' page strides are runtime arguments, and each job reads
its rows' records from the plan tensor through a control buffer. Every core's job list
is padded to 39 jobs, so the argument counts do not depend on the lengths. The program
is keyed by D, Dv, the scale, the numerics and its block sizes: a new set of lengths, or
a new total token count, compiles nothing. A lane job is up to `length x R` consecutive
query tile rows of one head and streams the key tiles its rows see.

## Numerics

As the dense kernel (fp32 accumulation, fidelity and exponential from Kohaku-TT's
numerics). Rel RMS against float64 attention of each sequence alone, HiFi2: 3e-3 to
6.5e-3 per call (bound 8e-3). A sequence's output is bit-identical whatever its
neighbours hold: their keys get P exactly 0.

Gradients against float64 autograd of each sequence alone, HiFi2, rel RMS per call:
dq and dk 9e-3 to 1.1e-2 (1.7e-2 for 50 sequences of 3 tokens), dv 6.6e-3 to 7.8e-3
(bound 2.5e-2). The log-sum-exp is within 2e-2 of float64 (it carries the bf16 rounding
of the scores). The dense backward's limit for inputs far from zero mean
([flash.md](flash.md), Limits) holds here too.

## Measured

Blackhole p150, one card, HiFi2, eager, synchronized per call, q / k / v as views:

| Case | per call |
|---|---|
| 8 x 201 tokens, 12 heads, D 64 | 1.05 - 1.10 ms |
| 150 ... 250 tokens x 8 (1608 in total), 12 heads, causal | 1.06 - 1.10 ms |
| 700 + 13 + 1500 tokens, 4 heads over 2, D 64 | 1.13 - 1.16 ms |
| first call of a new set of lengths | 2.4 - 3.1 ms (plan built and uploaded) |
| backward, 8 x 201 tokens, 12 heads, D 64 | 1.9 ms |
| backward, 700 + 13 + 1500 tokens, 4 heads over 2, causal | 1.9 ms |
| backward, 1 + 31 + 33 + 5 + 64 + 100 tokens, 2 heads | 1.7 ms |

These calls are host bound: 0.2 - 0.5 ms per program is its runtime arguments (rebuilt
per call: the lane plan and 39 jobs for each core) and its launch, the rest the Python
and dispatcher path of torch's custom op. The times moved by up to 2x in runs of ten or
more consecutive calls within one process (forward 2.0, 1.5 and 1.05 ms, backward 3.6,
2.6 and 1.7 ms for the same call, on a host shared with other jobs); the table holds the
lowest level, and the cause of the others was not established. With a key mask the
lowest levels are the same within 0.1 ms (a CPU mask adds its upload, a tt mask its
conversion ops). The dense kernel on the padded batch
`[8, 12, 201, 64]` takes 0.2 - 0.3 ms through `scaled_dot_product_attention` and 0.57 ms
for its backward.

## Limits

- bf16; D and Dv multiples of 32 up to 256.
- Logits (scale x score) must stay far below 5e5 in magnitude.
- Masks: the key mask `[Tk]` above only. A float mask must be finite or `-inf`: `+inf`
  and NaN are not rejected and give NaN rows.
- Causal with more queries than keys in a sequence is rejected (queries without keys).
- At most 39 jobs per core: about 39 x R query tile rows per core (key tile rows in the
  dK / dV program), which bounds the total at several hundred thousand tokens x heads;
  beyond that the call raises.
- The backward takes the forward's own output tensor (autograd passes it): the row
  statistics are found by it.
- A captured trace holds one set of lengths.
- Two steps run as loops on the readers' RISC-V cores. Zeroing the rows past the last
  token of a last tile (up to 31 x 32 stores per tile) is bounded: one tile row per
  tensor and job, and it keeps undefined padding out of the scores. Writing a key mask's
  values into the plan's mask tiles (1024 loads and conditional stores per mask tile, per
  row and head) grows with the mask tiles and can bound long sequences with a key mask;
  adding the mask row in the compute kernel would replace it.
- Equal lengths gain nothing over the padded dense call, and at these sizes the padded
  dense call is faster (host cost above); the saving is the padding of unequal lengths
  (score tiles follow the sum of Lq x Lk, not B x max^2).
