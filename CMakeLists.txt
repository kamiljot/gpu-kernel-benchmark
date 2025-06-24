cmake_minimum_required(VERSION 3.18)
project(gpu-kernel-benchmark LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source groups
file(GLOB_RECURSE CORE_CPP src/*.cpp)
list(REMOVE_ITEM CORE_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM CORE_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark_batch.cpp")

file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
file(GLOB_RECURSE HEADERS include/*.h include/*.cuh)

include_directories(include)

# Executable: main benchmark
add_executable(gpu_kernel_benchmark
    src/main.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
    ${HEADERS}
)

# Executable: batch benchmark
add_executable(gpu_kernel_batch
    src/benchmark_batch.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
    ${HEADERS}
)

# CUDA options
enable_language(CUDA)

set_target_properties(gpu_kernel_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

set_target_properties(gpu_kernel_batch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Windows-specific options
if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Output folders
set_target_properties(gpu_kernel_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(gpu_kernel_batch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
