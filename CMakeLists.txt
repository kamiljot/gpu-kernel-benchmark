cmake_minimum_required(VERSION 3.18)
project(gpu-kernel-benchmark LANGUAGES C CXX CUDA)

# C++ and CUDA settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Collect source files
file(GLOB_RECURSE SRC_CPP src/*.cpp)
file(GLOB_RECURSE SRC_CU src/*.cu)
file(GLOB_RECURSE HEADERS include/*.h include/*.cuh)

# Separate main files for different executables
list(REMOVE_ITEM SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark_batch.cpp")

# Include project headers
include_directories(include)

# Dynamically detect CUDA include path
if(WIN32 AND DEFINED ENV{CUDA_PATH})
    include_directories("$ENV{CUDA_PATH}/include")
endif()

# Executable: main benchmark
add_executable(gpu_kernel_benchmark
    src/main.cpp
    ${SRC_CPP}
    ${SRC_CU}
    ${HEADERS}
 "include/add.h" "include/sin_cos_pow_relu.h" "include/sqrt_log.h")

# Executable: batch benchmark
add_executable(gpu_kernel_batch
    src/benchmark_batch.cpp
    ${SRC_CPP}
    ${SRC_CU}
    ${HEADERS}
 "include/add.h" "include/sin_cos_pow_relu.h" "include/sqrt_log.h")

# Common CUDA options
enable_language(CUDA)
foreach(target IN ITEMS gpu_kernel_benchmark gpu_kernel_batch)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

# Windows-specific definitions
if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()