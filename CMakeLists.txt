# CMakeLists.txt
# @file    CMakeLists.txt
# @brief   CMake build script for gpu-kernel-benchmark (full registry setup & CUDA).
# @author  Kamil J.
# @date    2025-07-10

cmake_minimum_required(VERSION 3.15)
project(gpu-kernel-benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(include)

# Library with all core implementation
add_library(kernels
    src/kernels/add/add_kernel.cpp
    src/kernels/add/add_kernel_cuda.cu
    src/kernels/register_kernels.cpp

    src/backend/cpu_backend.cpp
    src/backend/cuda_backend.cpp
    src/backend/register_backends.cpp

    src/scenarios/dummy_scenario.cpp
    src/scenarios/register_scenarios.cpp

    # Utilities
    #src/utils/time_utils.cpp
    #src/utils/csv_logger.cpp
)

# Main CLI executable (benchmark runner)
add_executable(gpu-kernel-benchmark src/benchmark_runner.cpp)
target_link_libraries(gpu-kernel-benchmark kernels)

# Test executable (optional)
add_executable(registry_test tests/registry_test.cpp)
target_link_libraries(registry_test kernels)

# CUDA flags (optional, jeœli s¹ pliki .cu)
if (CMAKE_CUDA_COMPILER)
    set_target_properties(kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()