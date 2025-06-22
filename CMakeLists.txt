cmake_minimum_required(VERSION 3.10)
project(gpu_kernel_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

include_directories(include)

add_executable(gpu_kernel_benchmark
    src/main.cpp
    src/cpu_baseline.cpp
    src/gpu_kernel.cu
 "src/gpu_math.h")

target_link_libraries(gpu_kernel_benchmark cuda)

set_target_properties(gpu_kernel_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)
