# Experiments

Standalone proofs of concept: each script shows one behavior worth seeing on real
hardware and runs on its own with Kohaku-TT installed. They print error statistics from
`kohakutt.core.stats` (abs and rel error max / median / mean, cosine similarity,
non-finite count) against a float64 CPU result, next to the CPU result at the same
dtype, without pass / fail judgement.

The top-level script runs on device 0 (select a chip with `TT_VISIBLE_DEVICES`); the
scripts in `mesh/` open chips 0 and 1 as a 1x2 mesh and need two cards.

| Script | What it shows | Run |
|---|---|---|
| `attention_isolation.py` | TT-NN's SDPA mixes outputs across (batch, head) pairs once a call has many pairs: each pair of one call against the same pair computed alone, per layout and length ([TT-NN known issues](../docs/reference/ttnn-known-issues.md#sdpa-mixes-outputs-across-batch-head-pairs)) | `KTT_SDPA_MAX_UNITS=100000 python experiments/attention_isolation.py` (unsplit TT-NN SDPA; without the variable Kohaku-TT's split hides the defect) |
| `mesh/mesh_m0.py` | mesh basics: placed host transfers and shards round-trip bit for bit, automatic placement, the gather fallback, strict mode, RNG replicas, mesh memory totals ([mesh](../docs/concepts/mesh.md)) | `python experiments/mesh/mesh_m0.py` (`--cases <prefix>` for a subset; `KTT_FABRIC=0` for host-path conversions) |
| `mesh/mesh_trace.py` | captured traces containing collectives (all-gather, all-reduce, reduce-scatter) replay on new inputs bit for bit equal to eager ([trace](../docs/concepts/trace.md), [mesh](../docs/concepts/mesh.md#traces)) | `python experiments/mesh/mesh_trace.py` |
| `mesh/mesh_mlp.py` | transformer MLP blocks with column-parallel up and row-parallel down projections on two chips: the collectives per block, the output placement, and the result against one chip and the CPU | `python experiments/mesh/mesh_mlp.py` |

Per-op timing and precision live in the package: `python -m kohakutt.bench.ops`.
