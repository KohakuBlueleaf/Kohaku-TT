# Experiments

Standalone scripts behind findings in `docs/`. Each runs on device 0 (select a chip
with `TT_VISIBLE_DEVICES`) and reports error statistics with
`kohakutt.core.stats` (abs and rel error max / median / mean, cosine similarity)
without pass / fail judgement.

| script | measures | doc |
|---|---|---|
| `attention_isolation.py` | each (batch, head) pair of one SDPA call against the pair computed alone, per layout and length | `docs/reference/ttnn-issues.md` (SDPA) |

Per-op timing and precision are in the package: `python -m kohakutt.bench.ops`.
