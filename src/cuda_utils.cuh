#pragma once
#include <cuda_runtime.h>
#include <tuple>
#include <stdexcept> 
#include <cstdio>

// Macro for checking CUDA errors in host code.
#define CHECK_CUDA(call)                                                      \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

// Measures time taken to launch and complete a kernel using CUDA events
template<typename KernelFunc>
float measure_kernel_time(KernelFunc kernel_call) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    kernel_call();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return time_ms;
}

// Allocates device memory and copies host arrays a, b to device
inline std::tuple<float*, float*, float*> allocate_and_copy_to_device(const float* a, const float* b, int N) {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t size = N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    CHECK_CUDA(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    return { d_a, d_b, d_c };
}

// Copies device array d_c to host array c and frees all device memory
inline void copy_from_device_and_free(float* c, float* d_c, float* d_a, float* d_b, int N) {
    size_t size = N * sizeof(float);
    CHECK_CUDA(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// float4 variants

inline std::tuple<float4*, float4*, float4*> allocate_and_copy_to_device_float4(const float* a, const float* b, int N_vec4) {
    float4* d_a = nullptr;
    float4* d_b = nullptr;
    float4* d_c = nullptr;
    size_t size = N_vec4 * sizeof(float4);
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    CHECK_CUDA(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    return { d_a, d_b, d_c };
}

inline void copy_from_device_and_free_float4(float* c, float4* d_c, float4* d_a, float4* d_b, int N_vec4) {
    size_t size = N_vec4 * sizeof(float4);
    CHECK_CUDA(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}