# kohakutt_native: the torch "tt" device (PrivateUse1) over TT-NN tensors.
#
# Two translation-unit groups that never share headers: the bridge sees TT-NN only,
# the backend sees torch only (each bundles its own fmt).
cmake_minimum_required(VERSION 3.24)
project(kohakutt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Next to the package by default; a release build puts one build per torch elsewhere.
set(KOHAKUTT_NATIVE_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "library directory")
include(cmake/KohakuTT.cmake)
kohakutt_ttnn_flags()

# Device side (TT-NN / Metalium only, the build's own defines): core/device, each
# family's device_*.cpp and its own programs' ktt_*/program.cpp (their reader / compute
# / writer sources are JIT-compiled on the device). Torch side (torch only): core/torch,
# runtime and each family's aten_*.cpp. A file includes one side's headers, never both.
file(
    GLOB bridge_sources CONFIGURE_DEPENDS
    csrc/core/device/*.cpp csrc/core/mesh/device_*.cpp csrc/aten/*/device_*.cpp
    csrc/aten/*/ktt_*/program.cpp
)
# The C ABI of the tinygrad backend: the only default-visibility symbols (ktt_*).
list(APPEND bridge_sources csrc/py_bridge/capi.cpp)
file(
    GLOB backend_sources CONFIGURE_DEPENDS
    csrc/core/torch/*.cpp csrc/runtime/*.cpp csrc/aten/*/aten_*.cpp
)
# tt-train's fused training ops (host side; their kernels JIT-compile from the tt-metal
# tree): attention forward / backward.
set(TTML_ROOT "${TT_METAL_ROOT}/tt-train/sources/ttml")
file(
    GLOB ttml_sources CONFIGURE_DEPENDS
    ${TTML_ROOT}/metal/ops/sdpa_fw/*.cpp ${TTML_ROOT}/metal/ops/sdpa_fw/device/*.cpp
    ${TTML_ROOT}/metal/ops/sdpa_bw/*.cpp ${TTML_ROOT}/metal/ops/sdpa_bw/device/*.cpp
)
list(APPEND bridge_sources ${ttml_sources})

add_library(ktt_bridge OBJECT ${bridge_sources})
target_include_directories(ktt_bridge PRIVATE csrc include ${TTML_ROOT} ${TT_METAL_ROOT})
target_include_directories(ktt_bridge SYSTEM PRIVATE ${KOHAKUTT_TTNN_INCLUDES})
# TT-Metal's runtime headers (impl/, llrt/): the SoC descriptor behind multicast panels.
target_include_directories(
    ktt_bridge SYSTEM PRIVATE ${TT_METAL_ROOT}/tt_metal ${TT_METAL_ROOT}/tt_metal/hw/inc
)
target_compile_options(
    ktt_bridge PRIVATE ${KOHAKUTT_TTNN_DEFINES} -fPIC -O2 -fvisibility=hidden
)

# Backend: torch only.
kohakutt_find_torch()
add_library(
    _ktt_native SHARED
    ${backend_sources}
    $<TARGET_OBJECTS:ktt_bridge>)
target_include_directories(_ktt_native PRIVATE csrc include)
target_link_libraries(
    _ktt_native PRIVATE
    ${TORCH_LIBRARIES}
    ${KOHAKUTT_TT_METAL_LIBRARIES})
target_compile_options(_ktt_native PRIVATE -O2)
set_target_properties(
    _ktt_native PROPERTIES
    PREFIX ""
    BUILD_RPATH "${KOHAKUTT_TT_METAL_RPATH}"
    LIBRARY_OUTPUT_DIRECTORY "${KOHAKUTT_NATIVE_OUTPUT}")
