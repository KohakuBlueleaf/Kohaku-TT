# The release build image: tt-metal's build dependencies (its install_dependencies.sh at
# the pinned tag), CMake, ccache, uv with Python 3.13 / 3.14, auditwheel and twine.
# Built by packaging/ci/docker.sh; the build itself runs in it as packaging/ci/build.sh.
FROM ubuntu:24.04

ARG TT_METAL_PIN
ARG CMAKE_VERSION=4.0.2
LABEL org.opencontainers.image.title="kohakutt-build"

# --docker: no system sfpi (the tt-metal build fetches its own) and no CMake (below).
ADD --chmod=755 \
    https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_PIN}/install_dependencies.sh \
    /tmp/install_dependencies.sh
RUN /tmp/install_dependencies.sh --docker --no-distributed \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ccache file patchelf zip unzip \
    && wget -q -O /tmp/cmake.sh \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh" \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/cmake.sh /tmp/install_dependencies.sh

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin
RUN UV_NO_CACHE=1 uv python install 3.13 3.14 \
    && UV_NO_CACHE=1 uv tool install auditwheel \
    && UV_NO_CACHE=1 uv tool install twine \
    && chmod -R a+rX /opt/python /opt/uv-tools
# Jobs run as the host user: no Python downloads, caches on the bind-mounted cache.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-managed \
    CC=clang-20 \
    CXX=clang++-20
