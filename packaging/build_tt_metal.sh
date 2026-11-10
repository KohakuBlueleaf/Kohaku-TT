#!/usr/bin/env bash
# Clones and builds the pinned tt-metal (packaging/tt-metal.pin) into DIR.
# usage: build_tt_metal.sh DIR    (an existing build in DIR is kept;
# KTT_SKIP_DEPS=1 skips the system package step, which needs sudo)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=$1
pin=$(tr -d '[:space:]' < "$here/tt-metal.pin")

if [ ! -d "$dir/.git" ]; then
    git clone --depth 1 --branch "$pin" --recurse-submodules --shallow-submodules \
        https://github.com/tenstorrent/tt-metal.git "$dir"
fi
have=$(git -C "$dir" describe --tags --always)
[ "$have" = "$pin" ] || { echo "$dir holds tt-metal $have, the pin is $pin"; exit 1; }

# Compilers and system libraries: needed to build against the tree even when cached.
if [ "${KTT_SKIP_DEPS:-0}" != 1 ]; then
    sudo "$dir/install_dependencies.sh" --no-distributed
fi

# Without multi-host support: OpenMPI cannot be carried inside a wheel.
# Only what the wheel carries: the libraries and the generated kernel runtime files.
if [ ! -f "$dir/build/ttnn/_ttnncpp.so" ]; then
    (cd "$dir" && ./build_metal.sh --release --without-distributed --configure-only)
    ninja -C "$dir/build" ttnncpp hw_toolchain
fi
ls -la "$dir/build/ttnn/_ttnncpp.so" "$dir/build/tt_metal/libtt_metal.so"
