cmake_minimum_required(VERSION 3.18)
project(gpu-kernel-benchmark LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

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

# Enable testing
enable_testing()

# Test: add correctness
add_executable(test_add_correctness
    tests/test_add_correctness.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
    ${HEADERS}
)
set_target_properties(test_add_correctness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)
target_include_directories(test_add_correctness PRIVATE include)
target_link_libraries(test_add_correctness CUDA::cudart)
add_test(NAME AddCorrectnessTest COMMAND test_add_correctness)

# Test: sqrt_log correctness
add_executable(test_sqrt_log_correctness
    tests/test_sqrt_log_correctness.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
    ${HEADERS}
)
set_target_properties(test_sqrt_log_correctness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)
target_include_directories(test_sqrt_log_correctness PRIVATE include)
target_link_libraries(test_sqrt_log_correctness CUDA::cudart)
add_test(NAME SqrtLogCorrectnessTest COMMAND test_sqrt_log_correctness)

# Test: sin_cos_pow_relu correctness
add_executable(test_sin_cos_pow_relu_correctness
    tests/test_sin_cos_pow_relu_correctness.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
    ${HEADERS}
)
set_target_properties(test_sin_cos_pow_relu_correctness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)
target_include_directories(test_sin_cos_pow_relu_correctness PRIVATE include)
target_link_libraries(test_sin_cos_pow_relu_correctness CUDA::cudart)
add_test(NAME SinCosPowReluCorrectnessTest COMMAND test_sin_cos_pow_relu_correctness)