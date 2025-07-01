// Implements device memory allocation, deallocation, and copy routines for GPU arrays.

#include "gpu_memory_utils.h"

void allocate_and_copy(const float* a, const float* b, float** d_a, float** d_b, float** d_c, int N) {
    size_t size = N * sizeof(float);
    cudaMalloc(d_a, size);
    cudaMalloc(d_b, size);
    cudaMalloc(d_c, size);
    cudaMemcpy(*d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, b, size, cudaMemcpyHostToDevice);
}

void allocate_and_copy_vec4(const float* a, const float* b, float4** d_a4, float4** d_b4, float4** d_c4, int N_vec4) {
    size_t size = N_vec4 * sizeof(float4);
    cudaMalloc(d_a4, size);
    cudaMalloc(d_b4, size);
    cudaMalloc(d_c4, size);
    cudaMemcpy(*d_a4, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b4, b, size, cudaMemcpyHostToDevice);
}

void free_device(float* d_a, float* d_b, float* d_c) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
