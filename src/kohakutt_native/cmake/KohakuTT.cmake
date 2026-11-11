# Build helpers shared by kohakutt_native and Kohaku-TT extension libraries.
#
#   include(<kohakutt_native>/cmake/KohakuTT.cmake)
#   kohakutt_add_extension(my_ops DEVICE_SOURCES program.cpp TORCH_SOURCES op.cpp)
#
# Needs TT_METAL_ROOT (or TT_METAL_HOME) pointing at the tt-metal tree kohakutt was
# built against, and Python_EXECUTABLE with torch installed.

get_filename_component(KOHAKUTT_NATIVE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(NOT TT_METAL_ROOT)
    set(TT_METAL_ROOT "$ENV{TT_METAL_HOME}" CACHE PATH "tt-metal source tree")
endif()
if(NOT TT_METAL_ROOT)
    message(FATAL_ERROR "Set -DTT_METAL_ROOT=<tt-metal source tree> (or TT_METAL_HOME)")
endif()
set(TT_METAL_BUILD "${TT_METAL_ROOT}/build" CACHE PATH "tt-metal build directory")
set(
    KOHAKUTT_TT_METAL_LIBRARIES
    ${TT_METAL_BUILD}/ttnn/_ttnncpp.so
    ${TT_METAL_BUILD}/tt_metal/libtt_metal.so
    ${TT_METAL_BUILD}/tt_stl/libtt_stl.so
)
set(
    KOHAKUTT_TT_METAL_RPATH
    "${TT_METAL_BUILD}/ttnn;${TT_METAL_BUILD}/tt_metal;${TT_METAL_BUILD}/tt_stl;${TT_METAL_BUILD}/lib;${TT_METAL_BUILD}/tt_metal/third_party/umd/lib"
)

# TT-NN compile flags: the include directories and defines of one ttnn_core unity
# object in the tt-metal build, as KOHAKUTT_TTNN_INCLUDES / KOHAKUTT_TTNN_DEFINES.
function(kohakutt_ttnn_flags)
    execute_process(
        COMMAND ninja -C ${TT_METAL_BUILD} -t commands
                ttnn/CMakeFiles/ttnn_core.dir/Unity/unity_1_cxx.cxx.o
        OUTPUT_VARIABLE commands
        COMMAND_ERROR_IS_FATAL ANY)
    string(REGEX MATCH "[^\n]*clang\\+\\+[^\n]*unity_1_cxx[^\n]*" command "${commands}")
    string(REGEX MATCHALL "(-I|-isystem )[^ ]+" include_flags "${command}")
    string(REGEX MATCHALL "-D[^ ]+" defines "${command}")
    list(FILTER defines EXCLUDE REGEX "EXPORTS|TT_METAL_BUILD_TYPE")
    set(includes "")
    foreach(flag IN LISTS include_flags)
        string(REGEX REPLACE "^(-I|-isystem )" "" dir "${flag}")
        list(APPEND includes "${dir}")
    endforeach()
    list(LENGTH includes count)
    message(STATUS "TT-NN: ${count} include directories from ${TT_METAL_BUILD}")
    set(KOHAKUTT_TTNN_INCLUDES "${includes}" PARENT_SCOPE)
    set(KOHAKUTT_TTNN_DEFINES "${defines}" PARENT_SCOPE)
endfunction()

# Torch: find_package(Torch) with the prefix of Python_EXECUTABLE's torch.
macro(kohakutt_find_torch)
    if(NOT Python_EXECUTABLE)
        find_package(Python REQUIRED COMPONENTS Interpreter)
    endif()
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE kohakutt_torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND CMAKE_PREFIX_PATH "${kohakutt_torch_prefix}")
    find_package(Torch REQUIRED)
endmacro()

# An extension library `name` (a shared object loaded by kohakutt.extension.load):
# DEVICE_SOURCES see TT-NN / Metalium and ktt/extension/device.hpp only,
# TORCH_SOURCES see torch and ktt/extension/torch.hpp only.
function(kohakutt_add_extension name)
    cmake_parse_arguments(arg "" "" "DEVICE_SOURCES;TORCH_SOURCES" ${ARGN})
    kohakutt_ttnn_flags()
    kohakutt_find_torch()
    # kohakutt's library: a source tree's own build, else the wheel's for this torch.
    set(native "${KOHAKUTT_NATIVE_DIR}/_ktt_native.so")
    if(NOT EXISTS "${native}")
        set(series "${Torch_VERSION_MAJOR}.${Torch_VERSION_MINOR}")
        set(native "${KOHAKUTT_NATIVE_DIR}/torch${series}/_ktt_native.so")
    endif()
    if(NOT EXISTS "${native}")
        message(FATAL_ERROR "kohakutt has no native library for torch ${Torch_VERSION}")
    endif()
    get_filename_component(native_dir "${native}" DIRECTORY)
    add_library(${name}_device OBJECT ${arg_DEVICE_SOURCES})
    target_include_directories(${name}_device PRIVATE ${KOHAKUTT_NATIVE_DIR}/include)
    target_include_directories(${name}_device SYSTEM PRIVATE ${KOHAKUTT_TTNN_INCLUDES})
    target_compile_options(${name}_device PRIVATE ${KOHAKUTT_TTNN_DEFINES} -fPIC -O2)
    target_compile_features(${name}_device PRIVATE cxx_std_20)
    add_library(${name} SHARED ${arg_TORCH_SOURCES} $<TARGET_OBJECTS:${name}_device>)
    target_include_directories(${name} PRIVATE ${KOHAKUTT_NATIVE_DIR}/include)
    target_compile_features(${name} PRIVATE cxx_std_20)
    target_link_libraries(
        ${name} PRIVATE
        ${TORCH_LIBRARIES}
        ${native}
        ${KOHAKUTT_TT_METAL_LIBRARIES})
    set_target_properties(
        ${name} PROPERTIES
        PREFIX ""
        BUILD_RPATH "${native_dir};${KOHAKUTT_TT_METAL_RPATH}")
endfunction()
