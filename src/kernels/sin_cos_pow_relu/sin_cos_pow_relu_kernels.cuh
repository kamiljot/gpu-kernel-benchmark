#pragma once
#include <cuda_runtime.h>

// CUDA kernel using global memory
__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N);

// CUDA kernel using shared memory
__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N);

// CUDA kernel using float4 vectorized access
__global__ void sin_cos_pow_relu_float4_kernel(const float* a, const float* b, float* c, int N);