#!/usr/bin/env bash
# The release build image (packaging/ci/Dockerfile) on a build host with Docker.
# usage: docker.sh image             build the image (cached layers: a few seconds)
#        docker.sh run CMD [ARG...]  run CMD in it as the calling user, with the
#                                    repository at /work, KTT_CI_CACHE at /cache
# env: KTT_CI_CACHE (persistent cache directory, required by run), KTT_CI_CPUS (CPU
# limit and build parallelism, default 48), KTT_CI_MEMORY (memory limit, default none)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
pin=$(tr -d '[:space:]' < "$repo/packaging/tt-metal.pin")
image="kohakutt-build:${pin}"

build_image() {
    docker build --progress=plain --provenance=false --build-arg "TT_METAL_PIN=${pin}" \
        -t "$image" "$repo/packaging/ci"
    # Images of other pins, and the one a changed Dockerfile replaced.
    docker image ls kohakutt-build --format "{{.Repository}}:{{.Tag}}" \
        | { grep -vxF "$image" || true; } | xargs -r docker image rm
    docker image prune -f --filter "label=org.opencontainers.image.title=kohakutt-build"
}

case "${1:-}" in
image)
    build_image
    docker image ls "$image"
    ;;
run)
    shift
    cache=${KTT_CI_CACHE:?set KTT_CI_CACHE to a persistent cache directory on this host}
    cpus=${KTT_CI_CPUS:-48}
    mkdir -p "$cache"
    docker image inspect "$image" > /dev/null 2>&1 || build_image >&2
    name="kohakutt-build-$$"
    args=(
        --rm --init --name "$name" --cpus "$cpus" --user "$(id -u):$(id -g)"
        -e HOME=/tmp -e KTT_CACHE=/cache -e "KTT_JOBS=${cpus%.*}" -e GITHUB_ACTIONS
        -v "$repo:/work" -v "$cache:/cache" -w /work
    )
    if [ -n "${KTT_CI_MEMORY:-}" ]; then
        args+=(--memory "$KTT_CI_MEMORY")
    fi
    # A cancelled job stops its container too.
    trap 'docker rm -f "$name" > /dev/null 2>&1 || true' EXIT
    trap 'exit 143' INT TERM
    docker run "${args[@]}" "$image" "$@" &
    wait $!
    ;;
*)
    echo "usage: $0 image | run CMD [ARG...]" >&2
    exit 2
    ;;
esac
