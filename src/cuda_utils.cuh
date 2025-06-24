#pragma once
#include <cuda_runtime.h>
#include <tuple>
#include <stdexcept>  // for std::invalid_argument

// Measures time taken to launch and complete a kernel using CUDA events
template<typename KernelFunc>
float measure_kernel_time(KernelFunc kernel_call) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_call();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time_ms;
}

// Allocates device memory and copies host arrays a, b to device
inline std::tuple<float*, float*, float*> allocate_and_copy_to_device(const float* a, const float* b, int N) {
    float* d_a, * d_b, * d_c;
    size_t size = N * sizeof(float);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    return { d_a, d_b, d_c };
}

// Copies device array d_c to host array c and frees all device memory
inline void copy_from_device_and_free(float* c, float* d_c, float* d_a, float* d_b, int N) {
    size_t size = N * sizeof(float);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// float4 variants

inline std::tuple<float4*, float4*, float4*> allocate_and_copy_to_device_float4(const float* a, const float* b, int N_vec4) {
    float4* d_a, * d_b, * d_c;
    size_t size = N_vec4 * sizeof(float4);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    return { d_a, d_b, d_c };
}

inline void copy_from_device_and_free_float4(float* c, float4* d_c, float4* d_a, float4* d_b, int N_vec4) {
    size_t size = N_vec4 * sizeof(float4);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}