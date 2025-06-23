cmake_minimum_required(VERSION 3.10)
project(gpu_kernel_benchmark CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use dynamic runtime library (for MSVC)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Include directories
include_directories(
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Source files
add_executable(gpu_kernel_benchmark
    src/main.cpp
    src/cpu_baseline.cpp
    src/gpu_kernel.cu
    src/gpu_memory_utils.cu
    src/gpu_benchmark_utils.cu
    src/benchmark_utils.cpp
)

# Link CUDA
target_link_libraries(gpu_kernel_benchmark cuda)

# CUDA properties
set_target_properties(gpu_kernel_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)