#!/usr/bin/env bash
# The release build, run in the build image (docker.sh run bash packaging/ci/build.sh):
# tt-metal at the pin (kept in the cache), the native library once per supported torch,
# the repaired wheel in OUT, then an import of the wheel under each torch.
# usage: build.sh [OUT]   (env: KTT_CACHE cache directory, KTT_JOBS build jobs)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
out=$(realpath -m "${1:-dist}")
cache=${KTT_CACHE:?set KTT_CACHE to the cache directory}
jobs=${KTT_JOBS:-$(nproc)}
# One native library per torch series, built against the newest patch release.
torches=(2.10.0 2.11.0 2.12.1 2.13.0 2.14.1 2.15.0)
cpu_index=https://download.pytorch.org/whl/cpu

export UV_CACHE_DIR=$cache/uv CPM_SOURCE_CACHE=$cache/cpm KOHAKUTT_CACHE=$cache/kohakutt
export CCACHE_DIR=$cache/ccache CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-25G}
export CCACHE_COMPILERCHECK=content
mkdir -p "$cache/tt-metal" "$cache/venvs"
# One build at a time per cache directory.
exec 9> "$cache/build.lock"
flock 9

step() {
    [ "${GITHUB_ACTIONS:-}" = true ] && echo "::endgroup::" && echo "::group::$*"
    echo "== $* ($(date -u +%H:%M:%S), ${SECONDS}s)"
}
venv() {  # venv DIR PYTHON TORCH: a fresh environment with that CPU torch
    uv venv -q --clear -p "$2" "$1"
    uv pip install -q -p "$1/bin/python" --index-url "$cpu_index" "torch==$3+cpu" numpy
}
series() { echo "${1%.*}"; }
[ "${GITHUB_ACTIONS:-}" = true ] && echo "::group::start"
ccache -z > /dev/null

pin=$(tr -d '[:space:]' < packaging/tt-metal.pin)
step "tt-metal $pin"
key="$pin-$(sha256sum packaging/build_tt_metal.sh | cut -c1-12)"
tt=$cache/tt-metal/$key
KTT_SKIP_DEPS=1 KTT_CCACHE=1 KTT_JOBS=$jobs CCACHE_BASEDIR=$tt \
    bash packaging/build_tt_metal.sh "$tt"
touch "$tt"
# Trees of other pins or build scripts, unused for two weeks.
find "$cache/tt-metal" -mindepth 1 -maxdepth 1 -type d ! -name "$key" -mtime +14 \
    -exec rm -rf {} +

# build/lib is setuptools' staging directory: anything left there lands in the wheel.
rm -rf build/native-* build/natives build/lib build/dist-raw
mkdir -p build
natives=()
for torch in "${torches[@]}"; do
    s=$(series "$torch")
    step "native library for torch $torch"
    env_dir=$cache/venvs/build-torch$s
    venv "$env_dir" 3.13 "$torch"
    TT_METAL_HOME=$tt cmake -S src/kohakutt_native -B "build/native-$s" -G Ninja \
        -DPython_EXECUTABLE="$env_dir/bin/python" \
        -DKOHAKUTT_NATIVE_OUTPUT="$repo/build/natives/torch$s" \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > "build/cmake-$s.log"
    ninja -C "build/native-$s" -j "$jobs"
    natives+=(--native "$s=build/natives/torch$s/_ktt_native.so")
done
ccache -s

step "wheel"
newest=$cache/venvs/build-torch$(series "${torches[-1]}")
"$newest/bin/python" packaging/build_wheel.py --tt-metal "$tt" --out build/dist-raw \
    "${natives[@]}"
torch_lib=$("$newest/bin/python" -c "import torch, pathlib; \
print(pathlib.Path(torch.__file__).parent / 'lib')")
mkdir -p "$out"
rm -f "$out"/*.whl
LD_LIBRARY_PATH=$torch_lib auditwheel repair build/dist-raw/*.whl -w "$out" \
    --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so
twine check "$out"/*.whl
wheel=$(ls "$out"/*.whl)
ls -la "$out"

# No card here: install into clean environments, fetch the kernel compiler, load.
smoke=$cache/venvs/smoke
for case in "${torches[@]/#/3.13:}" "3.14:${torches[-1]}"; do
    py=${case%%:*} torch=${case#*:}
    step "smoke: Python $py, torch $torch"
    venv "$smoke" "$py" "$torch"
    uv pip install -q -p "$smoke/bin/python" "$wheel"
    (cd /tmp && "$smoke/bin/ktt" sfpi > /dev/null)
    (cd /tmp && "$smoke/bin/python" -c "import kohakutt, kohakutt_native, sys, torch; \
assert kohakutt_native.load(); \
print('loaded', kohakutt_native.LIBRARY.parent.name, 'torch', torch.__version__, \
'python', sys.version.split()[0])")
done
step "smoke: torch 2.9 is refused at import"
venv "$smoke" 3.13 2.9.1
uv pip install -q -p "$smoke/bin/python" --no-deps "$wheel"
(cd /tmp && "$smoke/bin/python" -c "
try:
    import kohakutt_native
except ImportError as error:
    print('refused:', error)
else:
    raise SystemExit('kohakutt_native imported under torch 2.9')")
step "done: $(basename "$wheel"), $(stat -c %s "$wheel") bytes"
[ "${GITHUB_ACTIONS:-}" = true ] && echo "::endgroup::"
true
