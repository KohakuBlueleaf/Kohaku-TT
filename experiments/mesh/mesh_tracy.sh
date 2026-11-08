#!/usr/bin/env bash
# Tracy host zones + device profiler of mesh_devprof.py into OUT:
#   OUT/trace.tracy, OUT/zones.csv (every host and device zone), OUT/.logs/profile_log_device.csv
# usage: experiments/mesh/mesh_tracy.sh OUT --cards 2 --dim 2048 --rows 1024
# TRACY_BIN: directory of tracy-capture and tracy-csvexport (default: on PATH);
# PYTHON: the interpreter with kohakutt (default: python); TT_METAL_RUNTIME_ROOT: the
# tt-metal tree (default: TT_METAL_HOME).
set -euo pipefail
OUT=$(realpath -m "$1"); shift
CAPTURE=${TRACY_BIN:+$TRACY_BIN/}tracy-capture
EXPORT=${TRACY_BIN:+$TRACY_BIN/}tracy-csvexport
PY=${PYTHON:-python}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME:?set TT_METAL_RUNTIME_ROOT}}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
"$CAPTURE" -o "$OUT/trace.tracy" -f > "$OUT/capture.log" 2>&1 &
CAP=$!
TRACY_NO_EXIT=1 TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SYNC=1 \
    TT_METAL_PROFILER_DIR="$OUT" "$PY" experiments/mesh/mesh_devprof.py "$@" > "$OUT/run.log" 2>&1
wait "$CAP"
"$EXPORT" -u "$OUT/trace.tracy" > "$OUT/zones.csv"
grep -E '^\{' "$OUT/run.log"
