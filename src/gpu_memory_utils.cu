#include "gpu_memory_utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

std::tuple<float*, float*, float*> allocate_and_copy_to_device(const float* a, const float* b, int N) {
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    return { d_a, d_b, d_c };
}

void copy_from_device_and_free(float* c, float* d_c, float* d_a, float* d_b, int N) {
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

std::tuple<float4*, float4*, float4*> allocate_and_copy_to_device_float4(const float* a, const float* b, int N_vec4) {
    float4* d_a4, * d_b4, * d_c4;
    cudaMalloc(&d_a4, N_vec4 * sizeof(float4));
    cudaMalloc(&d_b4, N_vec4 * sizeof(float4));
    cudaMalloc(&d_c4, N_vec4 * sizeof(float4));

    cudaMemcpy(d_a4, a, N_vec4 * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4, b, N_vec4 * sizeof(float4), cudaMemcpyHostToDevice);

    return { d_a4, d_b4, d_c4 };
}

void copy_from_device_and_free_float4(float* c, float4* d_c4, float4* d_a4, float4* d_b4, int N_vec4) {
    cudaMemcpy(c, d_c4, N_vec4 * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaFree(d_a4);
    cudaFree(d_b4);
    cudaFree(d_c4);
}
