// Device kernel declarations for sin_cos_pow_relu kernel (global, shared, float4).

#pragma once
#include <cuda_runtime.h>

// Global memory kernel.
__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N);

// Shared memory kernel.
__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N);

// float4 vectorized kernel.
__global__ void sin_cos_pow_relu_float4_kernel(const float4* a, const float4* b, float4* c, int N);

