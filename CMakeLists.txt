cmake_minimum_required(VERSION 3.18)
project(gpu_kernel_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

include_directories(
    src/
    src/core
    src/backend
    src/kernels
    src/profiler
    src/scenarios
)

# ---- Compile CUDA kernels as static libs ----
# Helper macro for a kernel dir
function(add_kernel_library KERNEL_NAME KERNEL_DIR)
    file(GLOB KERNEL_SOURCES
        "${KERNEL_DIR}/*.cpp"
        "${KERNEL_DIR}/*.cu"
    )
    add_library(${KERNEL_NAME} STATIC ${KERNEL_SOURCES})
    target_include_directories(${KERNEL_NAME} PUBLIC ${KERNEL_DIR})
    target_link_libraries(${KERNEL_NAME} PUBLIC CUDA::cudart)
endfunction()

# Add each kernel as a separate library
add_kernel_library(add_kernel src/kernels/add)
add_kernel_library(sin_cos_pow_relu_kernel src/kernels/sin_cos_pow_relu)
# Add further kernels here, e.g.:
# add_kernel_library(mul_kernel src/kernels/mul)

# ---- Core and non-kernel sources ----
file(GLOB_RECURSE CORE_SOURCES
    src/core/*.cpp
    src/backend/*.cpp
    src/profiler/*.cpp
    src/scenarios/*.cpp
    main.cpp
)

add_executable(gpu_kernel_benchmark ${CORE_SOURCES})

set_target_properties(gpu_kernel_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu_kernel_benchmark
    PRIVATE CUDA::cudart
    add_kernel
    sin_cos_pow_relu_kernel
    # Add more kernel libs here
)

add_executable(test_add_kernel test_add_kernel.cpp)
target_link_libraries(test_add_kernel PRIVATE add_kernel CUDA::cudart)

if (MSVC)
    target_compile_options(gpu_kernel_benchmark PRIVATE /W4 /WX)
else()
    target_compile_options(gpu_kernel_benchmark PRIVATE -Wall -Wextra -Werror)
endif()