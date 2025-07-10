# CMakeLists.txt
# @file    CMakeLists.txt
# @brief   CMake build script for gpu-kernel-benchmark (full registry setup & test target).
# @author  Kamil J.
# @date    2025-07-10

cmake_minimum_required(VERSION 3.15)
project(gpu-kernel-benchmark)

set(CMAKE_CXX_STANDARD 20)

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

# Main CLI executable
add_executable(gpu-kernel-benchmark src/main.cpp)
target_link_libraries(gpu-kernel-benchmark kernels)

# Test executable (optional, see below)
add_executable(registry_test tests/registry_test.cpp)
target_link_libraries(registry_test kernels)