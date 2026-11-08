# Experiments

Standalone scripts behind findings in `docs/`. They report error statistics with
`kohakutt.core.stats` (abs and rel error max / median / mean, cosine similarity,
non-finite count) against a float64 CPU ruler, without pass / fail judgement. Scripts
at the top level run on device 0 (select a chip with `TT_VISIBLE_DEVICES`); the scripts
in `mesh/` open chips 0 and 1 as a 1×2 mesh unless they take `--cards 1`.

| script | measures | doc |
|---|---|---|
| `attention_isolation.py` | each (batch, head) pair of one SDPA call against the pair computed alone, per layout and length | `docs/reference/ttnn-issues.md` (SDPA) |

## Mesh (`mesh/`)

| script | measures |
|---|---|
| `mesh_m0.py` | placement types, host transfers by placement, automatic placement at creation |
| `mesh_rules.py` | every op rule: output placement and conversions, values against 1 chip and float64 |
| `mesh_uneven.py` | reductions and K-split matmul over uneven shards (padding masked) |
| `mesh_rng.py` | Philox RNG at the global element index (`KTT_PHILOX=1`): sharded = 1 chip |
| `mesh_alltoall.py` | S(i) → S(j) by one all-to-all against all-gather + partition |
| `mesh_trace.py` | captured traces with collectives: every replay against eager, bit for bit |
| `mesh_mlp.py`, `mesh_bench.py`, `mesh_tflops.py` | MLP stacks, matmuls and models on 1 chip vs 2 chips |
| `mesh_devprof.py` | an MLP stack per weight placement (`--plan auto / megatron / dp`), timed, looped or replayed from a trace |
| `mesh_marks.py` | placement marks set in `__init__`, on 1 chip or 2 |
| `mesh_placements.py` | each op's input / output placements and conversions in one forward |
| `mesh_profile.py` | synchronized time per aten op, with its conversions |
| `mesh_tracy.sh`, `mesh_tracy_report.py`, `mesh_tracy_ops.py`, `mesh_devprof_report.py` | Tracy host zones and the device profiler on one clock: device busy vs host-bound idle, per-program timelines |
| `mesh_model.py`, `mesh_moss.py`, `mesh_train.py`, `mesh_large.py` | Qwen2.5, MOSS-Image DiT, a training step, a model larger than one chip |
| `mesh_migrate.py` | placement migration of persistent tensors (`KTT_MESH_MIGRATE=1`) |
| `mesh_oracle/` | placements and conversions against PyTorch DTensor on a fake process group |
| `host_overhead.py`, `trace_replay_check.py`, `coverage_compare.py` | host time per own kernel, trace replay against eager, coverage on 1 chip vs a mesh |

Per-op timing and precision are in the package: `python -m kohakutt.bench.ops`.
