# CMakeLists.txt
# @file    CMakeLists.txt
# @brief   CMake build script for gpu-kernel-benchmark (full registry setup & optional CUDA).
# @author  Kamil J.
# @date    2025-07-10

cmake_minimum_required(VERSION 3.15)
project(gpu-kernel-benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
option(USE_CUDA "Build with CUDA support" OFF)

# CUDA detection and setup
if (USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
        set(HAVE_CUDA TRUE)
        message(STATUS "CUDA support enabled.")
    else()
        message(WARNING "CUDA Toolkit not found! Building without CUDA support.")
        set(HAVE_CUDA FALSE)
    endif()
else()
    set(HAVE_CUDA FALSE)
endif()

include_directories(include)

# Library with all core implementation
add_library(kernels
    src/kernels/add/add_kernel.cpp
    src/kernels/register_kernels.cpp

    src/backend/cpu_backend.cpp
    src/backend/register_backends.cpp

    src/scenarios/dummy_scenario.cpp
    src/scenarios/register_scenarios.cpp
)

# CUDA sources only if CUDA is available
if (HAVE_CUDA)
    target_sources(kernels PRIVATE
        src/kernels/add/add_kernel_cuda.cu
        src/backend/cuda_backend.cpp
    )
    set_target_properties(kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    # KLUCZOWA LINIA — dodaje œcie¿ki nag³ówków CUDA
    target_include_directories(kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Main CLI executable (benchmark runner)
add_executable(gpu-kernel-benchmark src/benchmark_runner.cpp)
target_link_libraries(gpu-kernel-benchmark kernels)

# Test executable (optional)
add_executable(registry_test tests/registry_test.cpp)
target_link_libraries(registry_test kernels)
