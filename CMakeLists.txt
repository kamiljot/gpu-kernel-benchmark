cmake_minimum_required(VERSION 3.21)
project(gpu-kernel-benchmark LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Overridable CUDA arch list, np.:
# -DGPU_BENCHMARK_CUDA_ARCHITECTURES=86
# -DGPU_BENCHMARK_CUDA_ARCHITECTURES="75;86"
set(GPU_BENCHMARK_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures for all targets")
message(STATUS "CUDA architectures: ${GPU_BENCHMARK_CUDA_ARCHITECTURES}")

set(CORE_CPP
    src/benchmark_utils.cpp
    src/cpu_baseline.cpp
    src/input_generator.cpp
    src/kernel_dispatch.cpp
)

file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*/*_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*/*_launcher.cu"
)

function(configure_target target_name)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${GPU_BENCHMARK_CUDA_ARCHITECTURES}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    target_include_directories(${target_name} PRIVATE include)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
endfunction()

add_executable(gpu_kernel_benchmark
    src/main.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
)
configure_target(gpu_kernel_benchmark)

add_executable(gpu_kernel_batch
    src/benchmark_batch.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
)
configure_target(gpu_kernel_batch)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

enable_testing()

add_executable(test_add_correctness
    tests/test_add_correctness.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
)
configure_target(test_add_correctness)
add_test(NAME AddCorrectnessTest COMMAND test_add_correctness)

add_executable(test_sqrt_log_correctness
    tests/test_sqrt_log_correctness.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
)
configure_target(test_sqrt_log_correctness)
add_test(NAME SqrtLogCorrectnessTest COMMAND test_sqrt_log_correctness)

add_executable(test_sin_cos_pow_relu_correctness
    tests/test_sin_cos_pow_relu_correctness.cpp
    ${CORE_CPP}
    ${CUDA_SOURCES}
)
configure_target(test_sin_cos_pow_relu_correctness)
add_test(NAME SinCosPowReluCorrectnessTest COMMAND test_sin_cos_pow_relu_correctness)