---
title: Install
summary: Installing Kohaku-TT from PyPI or building it from a clone against your own tt-metal build, and checking the host with ktt doctor.
tags:
  - overview
  - install
---

# Install

A Tenstorrent Blackhole card (developed on P150A) with its kernel driver and firmware is
needed either way.

## From PyPI

Linux x86-64, glibc 2.39 or newer, Python 3.13 or 3.14:

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
pip install kohakutt          # --pre for the nightly builds
ktt sfpi                      # once: fetches the RISC-V kernel compiler into the cache
ktt doctor
```

The wheel carries the native layer, the tt-metal libraries it was built against (the
version in `packaging/tt-metal.pin`) and the kernel sources the card-side compiler
reads, so no tt-metal checkout is needed. It requires the PyTorch release it was built
for. Caches (the kernel compiler, generated kernel sources) go to `KOHAKUTT_CACHE`,
default `~/.cache/kohakutt`.

## From a clone

To develop, or against your own tt-metal build:

- A tt-metal source build: TT-NN and TT-Metalium, with the `tt-train` sources.
- Python 3.13, CMake 3.24+, Ninja, clang++-20; PyTorch 2.15 (the CPU wheel: the card is
  added as a PyTorch device).

```
uv venv -p 3.13
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
uv pip install -e ".[dev]"

export TT_METAL_HOME=/path/to/tt-metal TT_METAL_RUNTIME_ROOT=/path/to/tt-metal
CXX=clang++-20 cmake -S src/kohakutt_native -B build/native -G Ninja \
    -DPython_EXECUTABLE=$PWD/.venv/bin/python
ninja -C build/native                     # src/kohakutt_native/_ktt_native.so
ktt doctor
```

The native layer builds against the tt-metal tree it is pointed at: CMake reads TT-NN's
include directories and defines from that tree's own build, so the two always match.

| Extra | Adds |
|---|---|
| `dev` | pytest, ruff, black, blacken-docs, clang-format |

## Checking the host

`ktt doctor` is a read-only report of the host and every card on it: identity, firmware,
PCIe link against the expected one, clocks, power and temperature, with a warning per
problem. Other commands: [knobs, commands and tools](reference/knobs.md#commands).
