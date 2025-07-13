cmake_minimum_required(VERSION 3.18)
project(gpu-kernel-benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# CUDA is REQUIRED
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
add_definitions(-DUSE_CUDA)
message(STATUS "CUDA support required and enabled.")

include_directories(include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# --- CORE STATIC LIBRARY ---
add_library(kernels STATIC
    src/kernels/add/add_kernel.cpp
    src/kernels/register_kernels.cpp

    src/backend/cpu_backend.cpp
    src/backend/cuda_backend.cpp
    src/backend/register_backends.cpp

    src/scenarios/dummy_scenario.cpp
    src/scenarios/register_scenarios.cpp

    src/benchmark/benchmark_cpu.cpp 
    src/benchmark/benchmark_cuda.cpp
    src/benchmark/cli_args.cpp
    src/benchmark/generators.cpp
    src/benchmark/select_kernels.cpp
    src/benchmark/result_logger.cpp
    src/benchmark/benchmark_selection.cpp
    src/benchmark/benchmark_dispatch.cpp
    src/benchmark/dispatch_benchmarks.cpp
    src/benchmark/selectors.cpp
)

set_target_properties(kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# --- Main CLI executable (benchmark runner) ---
add_executable(gpu-kernel-benchmark
    src/benchmark_runner.cpp
    src/kernels/add/add_kernel_global.cu
)

target_link_libraries(gpu-kernel-benchmark kernels)

# --- Test executable (optional) ---
add_executable(registry_test
    tests/registry_test.cpp
    src/kernels/add/add_kernel_global.cu
)

target_link_libraries(registry_test kernels)

set_target_properties(gpu-kernel-benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(registry_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
