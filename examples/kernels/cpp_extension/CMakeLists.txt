# The SwiGLU extension library: swiglu_program.cpp (device side) + swiglu_op.cpp
# (torch side), built against an installed kohakutt.
#
#   cmake -S examples/kernels/cpp_extension -B build/swiglu_ext -G Ninja \
#       -DPython_EXECUTABLE=$(which python) -DTT_METAL_ROOT=<tt-metal tree>
#   ninja -C build/swiglu_ext
cmake_minimum_required(VERSION 3.24)
project(kohakutt_swiglu_extension LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import kohakutt.extension as e; print(e.cmake_module())"
    OUTPUT_VARIABLE kohakutt_cmake OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
include(${kohakutt_cmake})

kohakutt_add_extension(
    swiglu_ext
    DEVICE_SOURCES swiglu_program.cpp
    TORCH_SOURCES swiglu_op.cpp)
target_compile_definitions(
    swiglu_ext_device PRIVATE
    KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../metalium/kernels")
